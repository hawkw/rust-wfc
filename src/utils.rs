use ndarray::prelude::*;
use rand::distributions::{Range, IndependentSample};
use rand;
use bit_vec::BitVec;


pub fn rotate_90_clockwise<T: Copy>(image_data: &Array2<T>) -> Array2<T> {
    let mut output = image_data.t();
    output.invert_axis(Axis(1));
    output.to_owned()
}

pub fn masked_weighted_choice<T>(input: &[(T, usize)], mask: &BitVec) -> usize {
    /// Returns an index from the slice of (T, u) where u is the integer weight, i.e.
    /// [(1, 3), (2, 1), (3, 1)] returns 0 (the index of 1) with probability 3/5

    let total: usize =
        input.iter().map(|&(_, u)| u).zip(mask.iter()).filter(|&(_, m)| m).map(|(u, _)| u).sum();
    let between = Range::new(0, total);
    let mut rng = rand::thread_rng();
    let mut choice: usize = between.ind_sample(&mut rng);

    for ((index, u), mask) in input.iter().map(|&(_, u)| u).enumerate().zip(mask.iter()) {
        if mask {
            if choice < u {
                return index;
            }
            choice = choice.saturating_sub(u);
        } else {
            continue;
        }
    }
    unreachable!();
}

trait Maskable<T>: Iterator<Item=T> + Sized {
    fn masked<Mask>(self, Mask) -> Masked<Self, Mask::IntoIter>
    where Mask: IntoIterator<Item=bool>;
}

impl<I, T> Maskable<T> for I where I: Iterator<Item=T> {

    fn masked<Mask>(self, mask: Mask) -> Masked<Self, Mask::IntoIter>
    where Mask: IntoIterator<Item=bool> {
        Masked { items: self, mask: mask.into_iter() }
    }

}

pub struct Masked<I, M> {
    items: I
  , mask: M
}

impl<I, M> Iterator for Masked<I, M>
where I: Iterator
    , M: Iterator<Item=bool>
{
    type Item=I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.mask.next()
            .and_then(|m| if m == true {
                // if the current element in the mask is true, return the
                // corresponding item
                self.items.next()
            } else {
                // if the current element in the mask is false, skip that
                // element by advancing the items iterator past it
                let _ = self.items.next();
                // and skipping this iterator to the next item.
                self.next()
            })
    }
}
