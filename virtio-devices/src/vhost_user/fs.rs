// Copyright 2019 Intel Corporation. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io;
use std::os::unix::io::AsRawFd;
use std::result;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Barrier, Mutex};

use event_monitor::event;
use log::{error, info};
use seccompiler::SeccompAction;
use serde::{Deserialize, Serialize};
use serde_with::{Bytes, serde_as};
use vhost::vhost_user::message::{
    VhostUserMMap, VhostUserMMapFlags, VhostUserProtocolFeatures, VhostUserVirtioFeatures,
};
use vhost::vhost_user::{
    FrontendReqHandler, HandlerResult, VhostUserFrontend, VhostUserFrontendReqHandler,
};
use vm_device::UserspaceMapping;
use vm_memory::{ByteValued, GuestMemoryAtomic};
use vm_migration::protocol::MemoryRangeTable;
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};
use vmm_sys_util::eventfd::EventFd;

use super::vu_common_ctrl::VhostUserHandle;
use super::{DEFAULT_VIRTIO_FEATURES, Error, Result};
use crate::seccomp_filters::Thread;
use crate::thread_helper::spawn_virtio_thread;
use crate::vhost_user::{VhostUserCommon, VhostUserState};
use crate::{
    ActivateError, ActivateResult, GuestMemoryMmap, GuestRegionMmap, MmapRegion,
    VIRTIO_F_ACCESS_PLATFORM, VirtioCommon, VirtioDevice, VirtioDeviceType, VirtioSharedMemoryList,
};

const NUM_QUEUE_OFFSET: usize = 1;
const DEFAULT_QUEUE_NUMBER: usize = 2;

pub type State = VhostUserState<VirtioFsConfig>;

/// Handles SHMEM_MAP / SHMEM_UNMAP requests sent by virtiofsd over the
/// backend channel.
///
/// SHMEM_MAP carries a file fd (received via SCM_RIGHTS), a file offset, an
/// offset into the DAX cache, a length, and a writable flag. We `mmap` the
/// fd into the cache window with `MAP_FIXED`, replacing the original
/// `PROT_NONE` placeholder. SHMEM_UNMAP undoes this by re-overlaying the
/// region with an anonymous `PROT_NONE` mapping.
///
/// The handler holds an `Arc<MmapRegion>` so the cache stays alive while
/// the daemon may still issue requests against it.
struct BackendReqHandler {
    mapping: Arc<MmapRegion>,
}

impl BackendReqHandler {
    fn check_bounds(&self, shm_offset: u64, len: u64) -> io::Result<*mut u8> {
        let off: usize = shm_offset
            .try_into()
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        let len_usz: usize = len
            .try_into()
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        let end = off
            .checked_add(len_usz)
            .ok_or_else(|| io::Error::from_raw_os_error(libc::EINVAL))?;
        if end > self.mapping.size() {
            return Err(io::Error::from_raw_os_error(libc::EINVAL));
        }
        // SAFETY: `off` is within `mapping.size()`, so this pointer is within
        // the region we own.
        Ok(unsafe { self.mapping.as_ptr().add(off) })
    }
}

impl VhostUserFrontendReqHandler for BackendReqHandler {
    fn shmem_map(&self, req: &VhostUserMMap, fd: &dyn AsRawFd) -> HandlerResult<u64> {
        let target = self.check_bounds(req.shm_offset, req.len)?;
        let flags = VhostUserMMapFlags::from_bits_truncate(req.flags);
        let prot = if flags.contains(VhostUserMMapFlags::WRITABLE) {
            libc::PROT_READ | libc::PROT_WRITE
        } else {
            libc::PROT_READ
        };
        // SAFETY: `target` is within the cache region (bounds-checked above);
        // MAP_FIXED replaces the previous PROT_NONE placeholder atomically.
        // The fd is owned by the caller; MAP_SHARED makes the kernel hold its
        // own reference to the underlying file, so dropping the fd after this
        // call is safe.
        let ret = unsafe {
            libc::mmap(
                target as *mut libc::c_void,
                req.len as usize,
                prot,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd.as_raw_fd(),
                req.fd_offset as libc::off_t,
            )
        };
        if ret == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(0)
    }

    fn shmem_unmap(&self, req: &VhostUserMMap) -> HandlerResult<u64> {
        let target = self.check_bounds(req.shm_offset, req.len)?;
        // SAFETY: same bounds invariant as shmem_map. Overlaying with an
        // anonymous PROT_NONE region releases the prior file-backed mapping
        // and restores the placeholder semantics: any guest access faults
        // until the next SHMEM_MAP.
        let ret = unsafe {
            libc::mmap(
                target as *mut libc::c_void,
                req.len as usize,
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if ret == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(0)
    }
}

pub const VIRTIO_FS_TAG_LEN: usize = 36;
#[serde_as]
#[derive(Copy, Clone, Serialize, Deserialize)]
#[repr(C, packed)]
pub struct VirtioFsConfig {
    #[serde_as(as = "Bytes")]
    pub tag: [u8; VIRTIO_FS_TAG_LEN],
    pub num_request_queues: u32,
}

impl Default for VirtioFsConfig {
    fn default() -> Self {
        VirtioFsConfig {
            tag: [0; VIRTIO_FS_TAG_LEN],
            num_request_queues: 0,
        }
    }
}

// SAFETY: only a series of integers
unsafe impl ByteValued for VirtioFsConfig {}

pub struct Fs {
    vu_common: VhostUserCommon,
    id: String,
    config: VirtioFsConfig,
    /// DAX shared-memory window. The `Arc<MmapRegion>` inside the list keeps
    /// the host-side anonymous mapping alive for the lifetime of the device;
    /// SHMEM_MAP requests from the daemon mmap file pages into it.
    cache: Option<VirtioSharedMemoryList>,
    seccomp_action: SeccompAction,
    guest_memory: Option<GuestMemoryAtomic<GuestMemoryMmap>>,
    exit_evt: EventFd,
    access_platform_enabled: bool,
}

impl Fs {
    /// Create a new virtio-fs device.
    ///
    /// When `dax_enabled` is true, the device additionally negotiates the
    /// `BACKEND_REQ | BACKEND_SEND_FD | SHMEM` vhost-user protocol features
    /// and queries the daemon for the size of the DAX shared-memory window
    /// via `GET_SHMEM_CONFIG`. The window size is returned as the second
    /// element of the tuple; the caller is expected to allocate the cache
    /// and pass it back via [`Fs::set_cache`] before the device is activated.
    ///
    /// Returns `Err(Error::DaxNotSupported)` if the caller asked for DAX but
    /// the daemon does not advertise the `SHMEM` protocol feature.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: String,
        path: &str,
        tag: &str,
        req_num_queues: usize,
        queue_size: u16,
        dax_enabled: bool,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        access_platform_enabled: bool,
        state: Option<State>,
    ) -> Result<(Fs, Option<u64>)> {
        // Calculate the actual number of queues needed.
        let num_queues = NUM_QUEUE_OFFSET + req_num_queues;

        // Connect to the vhost-user socket.
        let mut vu =
            VhostUserHandle::connect_vhost_user(false, path, num_queues as u64, false, None)?;

        // Size of the DAX shared-memory window the daemon wants us to allocate.
        // Stays `None` for snapshot restore and for non-DAX devices.
        let mut dax_cache_size: Option<u64> = None;

        let (
            avail_features,
            acked_features,
            acked_protocol_features,
            vu_num_queues,
            config,
            paused,
            vring_bases,
        ) = if let Some(state) = state {
            info!("Restoring vhost-user-fs {id}");

            vu.set_protocol_features_vhost_user(
                state.acked_features,
                state.acked_protocol_features,
            )?;

            vu.restore_state(&state)?;

            (
                state.avail_features,
                state.acked_features,
                state.acked_protocol_features,
                state.vu_num_queues,
                state.config,
                true,
                state.vring_bases,
            )
        } else {
            // Filling device and vring features VMM supports.
            let avail_features = DEFAULT_VIRTIO_FEATURES;

            let mut avail_protocol_features = VhostUserProtocolFeatures::MQ
                | VhostUserProtocolFeatures::CONFIGURE_MEM_SLOTS
                | VhostUserProtocolFeatures::REPLY_ACK
                | VhostUserProtocolFeatures::INFLIGHT_SHMFD
                | VhostUserProtocolFeatures::LOG_SHMFD
                | VhostUserProtocolFeatures::DEVICE_STATE;

            // DAX requires the backend to send SHMEM_MAP/UNMAP messages over
            // the backend channel; SHMEM gates GET_SHMEM_CONFIG and those
            // backend requests. We only ask for these if the user enabled DAX.
            if dax_enabled {
                avail_protocol_features |= VhostUserProtocolFeatures::BACKEND_REQ
                    | VhostUserProtocolFeatures::BACKEND_SEND_FD
                    | VhostUserProtocolFeatures::SHMEM;
            }

            let (acked_features, acked_protocol_features) =
                vu.negotiate_features_vhost_user(avail_features, avail_protocol_features)?;

            // If the user explicitly asked for DAX but the daemon didn't ack
            // SHMEM, fail loudly at startup instead of silently degrading.
            if dax_enabled && (acked_protocol_features & VhostUserProtocolFeatures::SHMEM.bits()) == 0
            {
                return Err(Error::DaxNotSupported);
            }

            // Learn the cache window size from the daemon. virtio-fs always uses
            // a single region; anything else is a protocol violation.
            if acked_protocol_features & VhostUserProtocolFeatures::SHMEM.bits() != 0 {
                let shmem_config = vu
                    .socket_handle()
                    .get_shmem_config()
                    .map_err(Error::VhostUserGetShmemConfig)?;
                if shmem_config.nregions != 1 {
                    return Err(Error::FsUnexpectedShmemRegions(shmem_config.nregions));
                }
                dax_cache_size = Some(shmem_config.memory_sizes[0]);
            }

            let backend_num_queues =
                if acked_protocol_features & VhostUserProtocolFeatures::MQ.bits() != 0 {
                    vu.socket_handle()
                        .get_queue_num()
                        .map_err(Error::VhostUserGetQueueMaxNum)? as usize
                } else {
                    DEFAULT_QUEUE_NUMBER
                };

            if num_queues > backend_num_queues {
                error!(
                    "vhost-user-fs requested too many queues ({num_queues}) since the backend only supports {backend_num_queues}\n"
                );
                return Err(Error::BadQueueNum);
            }

            // Create virtio-fs device configuration.
            let mut config = VirtioFsConfig::default();
            let tag_bytes_slice = tag.as_bytes();
            let len = if tag_bytes_slice.len() < config.tag.len() {
                tag_bytes_slice.len()
            } else {
                config.tag.len()
            };
            config.tag[..len].copy_from_slice(tag_bytes_slice[..len].as_ref());
            config.num_request_queues = req_num_queues as u32;

            (
                acked_features,
                // If part of the available features that have been acked, the
                // PROTOCOL_FEATURES bit must be already set through the VIRTIO
                // acked features as we know the guest would never ack it, thus
                // the feature would be lost.
                acked_features & VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits(),
                acked_protocol_features,
                num_queues,
                config,
                false,
                None,
            )
        };

        let fs = Fs {
            vu_common: VhostUserCommon {
                virtio_common: VirtioCommon {
                    device_type: VirtioDeviceType::Fs as u32,
                    avail_features,
                    acked_features,
                    queue_sizes: vec![queue_size; num_queues],
                    paused_sync: Some(Arc::new(Barrier::new(2))),
                    min_queues: 1,
                    paused: Arc::new(AtomicBool::new(paused)),
                    ..Default::default()
                },
                vu: Some(Arc::new(Mutex::new(vu))),
                acked_protocol_features,
                socket_path: path.to_string(),
                vu_num_queues,
                vring_bases,
                ..Default::default()
            },
            id,
            config,
            cache: None,
            seccomp_action,
            guest_memory: None,
            exit_evt,
            access_platform_enabled,
        };

        Ok((fs, dax_cache_size))
    }

    /// Install the DAX shared-memory cache after construction.
    ///
    /// Called by the device manager once it has allocated guest physical
    /// address space and an anonymous host mapping of the size declared by
    /// the daemon (see the second tuple element returned by [`Fs::new`]).
    /// Must be invoked before [`Fs::activate`] for DAX to take effect.
    pub fn set_cache(&mut self, cache: VirtioSharedMemoryList) {
        self.cache = Some(cache);
    }

    fn state(&self) -> std::result::Result<State, MigratableError> {
        self.vu_common.state(self.config)
    }
}

impl Drop for Fs {
    fn drop(&mut self) {
        self.vu_common.shutdown();
    }
}

impl VirtioDevice for Fs {
    fn device_type(&self) -> u32 {
        self.vu_common.virtio_common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.vu_common.virtio_common.queue_sizes
    }

    fn features(&self) -> u64 {
        let mut features = self.vu_common.virtio_common.avail_features;
        if self.access_platform_enabled {
            features |= 1u64 << VIRTIO_F_ACCESS_PLATFORM;
        }
        features
    }

    fn ack_features(&mut self, value: u64) {
        self.vu_common.virtio_common.ack_features(value);
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        self.read_config_from_slice(self.config.as_slice(), offset, data);
    }

    fn activate(&mut self, context: crate::device::ActivationContext) -> ActivateResult {
        let crate::device::ActivationContext {
            mem,
            interrupt_cb,
            queues,
            device_status,
        } = context;
        self.vu_common
            .virtio_common
            .activate(&queues, interrupt_cb.clone())?;
        self.guest_memory = Some(mem.clone());

        // When DAX is enabled the cache is populated by the device manager
        // before activation. We install a backend-channel handler that owns
        // a reference to the cache region and translates SHMEM_MAP/UNMAP
        // requests into mmap calls into that window.
        let backend_req_handler: Option<FrontendReqHandler<BackendReqHandler>> =
            if let Some(shm_list) = self.cache.as_ref() {
                let handler = Arc::new(BackendReqHandler {
                    mapping: shm_list.mapping.clone(),
                });
                let mut req_handler = FrontendReqHandler::new(handler).map_err(|e| {
                    ActivateError::VhostUserFsSetup(Error::FrontendReqHandlerCreation(e))
                })?;
                // REPLY_ACK ensures the daemon waits for our mmap to complete
                // before replying to the guest's FUSE_SETUPMAPPING request; the
                // guest's page fault must not resume against still-PROT_NONE
                // memory.
                if self.vu_common.acked_protocol_features
                    & VhostUserProtocolFeatures::REPLY_ACK.bits()
                    != 0
                {
                    req_handler.set_reply_ack_flag(true);
                }
                Some(req_handler)
            } else {
                None
            };
        // Run a dedicated thread for handling potential reconnections with
        // the backend.
        let (kill_evt, pause_evt) = self.vu_common.virtio_common.dup_eventfds();

        let mut handler = self.vu_common.activate(
            mem,
            &queues,
            interrupt_cb.clone(),
            self.vu_common.virtio_common.acked_features,
            backend_req_handler,
            kill_evt,
            pause_evt,
        )?;

        let paused = self.vu_common.virtio_common.paused.clone();
        let paused_sync = self.vu_common.virtio_common.paused_sync.clone();

        let mut epoll_threads = Vec::new();
        spawn_virtio_thread(
            &self.id,
            &self.seccomp_action,
            Thread::VirtioVhostFs,
            &mut epoll_threads,
            &self.exit_evt,
            device_status.clone(),
            interrupt_cb.clone(),
            move || handler.run(&paused, paused_sync.as_ref().unwrap()),
        )?;
        self.vu_common.epoll_thread = Some(epoll_threads.remove(0));

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }

    fn reset(&mut self) {
        self.vu_common.reset(&self.id);
    }

    fn shutdown(&mut self) {
        self.vu_common.shutdown();
    }

    fn get_shm_regions(&self) -> Option<VirtioSharedMemoryList> {
        self.cache.clone()
    }

    fn set_shm_regions(
        &mut self,
        shm_regions: VirtioSharedMemoryList,
    ) -> std::result::Result<(), crate::Error> {
        if self.cache.is_some() {
            self.cache = Some(shm_regions);
            Ok(())
        } else {
            Err(crate::Error::SetShmRegionsNotSupported)
        }
    }

    fn add_memory_region(
        &mut self,
        region: &Arc<GuestRegionMmap>,
    ) -> std::result::Result<(), crate::Error> {
        self.vu_common.add_memory_region(&self.guest_memory, region)
    }

    fn userspace_mappings(&self) -> Vec<UserspaceMapping> {
        let mut mappings = Vec::new();
        if let Some(cache) = self.cache.as_ref() {
            mappings.push(UserspaceMapping {
                mem_slot: cache.mem_slot,
                addr: cache.addr,
                mapping: cache.mapping.clone(),
                mergeable: false,
            });
        }

        mappings
    }
}

impl Pausable for Fs {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.pause()?;
        self.vu_common.virtio_common.pause()
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.virtio_common.resume()?;

        if let Some(epoll_thread) = &self.vu_common.epoll_thread {
            epoll_thread.thread().unpark();
        }

        self.vu_common.resume()
    }
}

impl Snapshottable for Fs {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn snapshot(&mut self) -> std::result::Result<Snapshot, MigratableError> {
        self.vu_common.snapshot(&self.state()?)
    }
}
impl Transportable for Fs {}

impl Migratable for Fs {
    fn start_dirty_log(&mut self) -> std::result::Result<(), MigratableError> {
        self.vu_common.start_dirty_log(&self.guest_memory)
    }

    fn stop_dirty_log(&mut self) -> std::result::Result<(), MigratableError> {
        self.vu_common.stop_dirty_log()
    }

    fn dirty_log(&mut self) -> std::result::Result<MemoryRangeTable, MigratableError> {
        self.vu_common.dirty_log(&self.guest_memory)
    }

    fn start_migration(&mut self) -> std::result::Result<(), MigratableError> {
        self.vu_common.start_migration()
    }

    fn complete_migration(&mut self) -> std::result::Result<(), MigratableError> {
        self.vu_common.complete_migration()
    }
}
