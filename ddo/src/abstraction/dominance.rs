use std::cmp::Ordering;

pub trait Dominance {
    type State;
    type Key;

    /// Takes a state and returns a key that maps it to comparable states
    fn get_key(&self, state: &Self::State) -> Option<Self::Key>;

    /// Returns the number of dimensions that capture the value of a state
    fn nb_value_dimensions(&self) -> usize;

    /// Returns the i-th component of the value associated with the given state
    /// Greater is better for the dominance check
    fn get_value_at(&self, state: &Self::State, i: usize) -> isize;

    /// Checks whether there is a dominance relation between the two states, given the values
    /// provided by the function get_value_at evaluated for all i in 0..self.nb_value_dimensions()
    /// Note: the states are assumed to have the same key, otherwise they are not comparable for dominance
    fn partial_cmp(&self, a: &Self::State, b: &Self::State) -> Option<Ordering> {
        let mut ordering = Ordering::Equal;
        for i in 0..self.nb_value_dimensions() {
            let val_a = self.get_value_at(a, i);
            let val_b = self.get_value_at(b, i);
            
            if val_a < val_b {
                if ordering == Ordering::Greater {
                    return None;
                } else if ordering == Ordering::Equal {
                    ordering = Ordering::Less;
                }
            } else if val_a > val_a {
                if ordering == Ordering::Less {
                    return None;
                } else if ordering == Ordering::Equal {
                    ordering = Ordering::Greater;
                }
            }
        }
        Some(ordering)
    }

    /// Comparator to order states by increasing value, regardless of their key
    fn cmp(&self, a: &Self::State, b: &Self::State) -> Ordering {
        for i in 0..self.nb_value_dimensions() {
            let val_a = self.get_value_at(a, i);
            let val_b = self.get_value_at(b, i);
            
            if val_a < val_b {
                return Ordering::Less;
            } else if val_a > val_b {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }

}

pub trait DominanceChecker {
    type State;
    
    /// Returns true if the state is dominated by a stored one
    /// And insert the (key, value) pair otherwise
    fn is_dominated_or_insert(&self, state: &Self::State) -> bool;

    /// Comparator to order states by increasing value, regardless of their key
    fn cmp(&self, a: &Self::State, b: &Self::State) -> Ordering;
    
}