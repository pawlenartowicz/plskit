// Pyodide builds have no threads: nothing can wake a waiter, and the crate's
// contract allows `wait` to return spuriously, so it returns at once.
use core::sync::atomic::AtomicU32;

#[inline]
pub fn wait(_a: &AtomicU32, _expected: u32) {
    core::hint::spin_loop();
}

#[inline]
pub fn wake_one(_ptr: *const AtomicU32) {}

#[inline]
pub fn wake_all(_ptr: *const AtomicU32) {}
