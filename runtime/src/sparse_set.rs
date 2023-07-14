//! Provides an implementation of a SparseSet as an alternative to HashSets.

extern crate alloc;
use alloc::{vec, vec::Vec};

pub struct SparseSet {
    elems: usize,
    dense: Vec<Option<usize>>,
    sparse: Vec<usize>,
}

impl SparseSet {
    /// Initializes a new set of taking a value representing the maximum size
    /// of the set.
    #[must_use]
    pub fn new(max_len: usize) -> Self {
        Self {
            elems: 0,
            // initialize 0th index to non-zero so it doesn't match as a set
            // member by default.
            dense: vec![],
            sparse: vec![0; max_len],
        }
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.elems == 0
    }

    /// Returns the number of elements that the set can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.sparse.capacity()
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        self.elems
    }

    /// Inserts a value into the set.
    pub fn insert(&mut self, val: usize) {
        if self.contains(&val) {
            return;
        }

        if self.sparse.capacity() <= val {
            // double the size.
            self.resize(val * 2)
        }

        let dense_len = self.dense.len();
        let sparse_idx = self.sparse[val];
        if dense_len > sparse_idx && self.dense[sparse_idx].is_none() {
            self.dense[sparse_idx] = Some(val)
        } else {
            self.dense.push(Some(val));
            self.sparse[val] = dense_len;
        }

        self.elems += 1;
    }

    /// Inserts a value into the set.
    ///
    /// This is the unchecked alternative to `insert`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The value must not exceed the max length of the set;
    ///
    /// Failing that will cause a panic.
    pub unsafe fn insert_unchecked(&mut self, val: usize) {
        let dense_len = self.dense.len();
        let sparse_idx = self.sparse[val];
        if dense_len > sparse_idx && self.dense[sparse_idx].is_none() {
            self.dense[sparse_idx] = Some(val)
        } else {
            self.dense.push(Some(val));
            self.sparse[val] = dense_len;
        }

        self.elems += 1
    }

    /// Returns `true` if the set contains a value.
    pub fn contains(&self, val: &usize) -> bool {
        self.sparse
            .get(*val)
            .map(|&dense_idx| self.dense.get(dense_idx) == Some(&Some(*val)))
            // if none, the bounds of the set are exceeded and thus doesn't
            // contain the value.
            .unwrap_or(false)
    }

    pub fn remove(&mut self, val: &usize) -> bool {
        if self.contains(val) {
            // safety guaranteed by above contains check
            unsafe { self.remove_unchecked(val) };
            true
        } else {
            false
        }
    }

    /// Removes a value into the set.
    ///
    /// This is the unchecked alternative to `remove`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The value must not exceed the max length of the set;
    /// * The value must be defined in the set.
    ///
    /// Failing that will cause a panic.
    unsafe fn remove_unchecked(&mut self, val: &usize) {
        let dense_idx = self.sparse[*val];
        self.dense[dense_idx] = None;
        self.elems -= 1;
    }

    /// Clears the set, removing all values.
    pub fn clear(&mut self) {
        self.elems = 0;
        self.dense.clear();
    }

    fn resize(&mut self, new_len: usize) {
        self.sparse.resize_with(new_len, || 0)
    }
}

impl core::fmt::Debug for SparseSet {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "SparseSet({:?}", &self.dense)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_cause_resize_on_insert_with_bound_exceed() {
        let mut set = SparseSet::new(0);

        // set capacity is equivalent to provided max_len.
        assert!(set.capacity() == 0);

        set.insert(10);

        // after insert capacity is 2x largest insert value.
        assert!(set.capacity() == 20);
    }

    #[test]
    fn should_clear_value_on_delete() {
        let mut set = SparseSet::new(0);

        set.insert(1);
        assert!(set.contains(&1));
        assert_eq!(1, set.len());

        set.remove(&1);
        assert!(!set.contains(&1));
        assert_eq!(0, set.len());
    }

    #[test]
    fn should_reuse_slots_on_delete() {
        let mut set = SparseSet::new(0);

        set.insert(1);
        assert!(set.contains(&1));
        assert_eq!(1, set.len());

        // assert value was cleared from set
        set.remove(&1);
        assert!(!set.contains(&1));
        assert!(set.dense.len() == 1 && set.dense[0].is_none());

        // Re-add the previous value and assert slot is reused.
        set.insert(1);
        assert!(set.contains(&1));
        assert!(set.dense.len() == 1 && set.dense[0] == Some(1));
    }
}
