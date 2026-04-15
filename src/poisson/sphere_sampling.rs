use std::collections::HashSet;
use std::f32::consts::TAU;

use glam::{USizeVec3, Vec3};
use itertools::Itertools;
use rand::distr::{Distribution, Uniform};
use rand::{Rng, seq::IteratorRandom};

use crate::poisson::octree::Octree;

struct SphereShellDistribution {
    center: Vec3,
    r: f32,
}
impl SphereShellDistribution {
    fn new(center: Vec3, r: f32) -> Self {
        Self { center, r }
    }
}
impl Default for SphereShellDistribution {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            r: 1.0,
        }
    }
}
impl Distribution<Vec3> for SphereShellDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec3 {
        let r = (7.0 * rng.random::<f32>() + 1.0).cbrt() * self.r;
        let a1 = rng.random::<f32>() * TAU;
        let a2 = rng.random::<f32>() * TAU;
        Vec3::X.rotate_y(a1).rotate_z(a2) * r + self.center
    }
}

fn check_constraints(
    point: Vec3,
    r: f32,
    min: Vec3,
    max: Vec3,
    existing: &Octree,
    depth: usize,
) -> bool {
    if !((min.x..max.x).contains(&point.x)
        && (min.y..max.y).contains(&point.y)
        && (min.z..max.z).contains(&point.z))
    {
        return false;
    }

    let node_index =
        ((point - existing.pos()) / existing.size() * (1 << depth) as f32).as_usizevec3();

    let offsets = (0..3)
        .cartesian_product(0..3)
        .cartesian_product(0..3)
        .map(|((x, y), z)| USizeVec3::new(x, y, z));

    let points = offsets
        .map(|x| x + node_index)
        .filter(|x| x.x != 0 && x.y != 0 && x.z != 0)
        .map(|x| x - 1)
        .filter(|x| {
            let max = 1 << depth;
            x.x < max && x.y < max && x.z < max
        })
        .flat_map(|x| existing.get_node(x.x, x.y, x.z, depth).iter());

    for p in points {
        if point.distance_squared(p) < r.powi(2) {
            return false;
        }
    }
    true
}
fn try_sample_point(
    rng: &mut impl Rng,
    seed: Vec3,
    min: Vec3,
    max: Vec3,
    existing: &Octree,
) -> Option<Vec3> {
    for _ in 0..5 {
        let r = 1.0;
        let depth = (existing.size() / r).log2() as usize;
        let new = rng.sample(SphereShellDistribution::new(seed, r));
        if check_constraints(new, r, min, max, existing, depth) {
            return Some(new);
        }
    }
    None
}
pub fn sample_points(rng: &mut impl Rng, mut octree: Octree) -> Vec<Vec3> {
    let mut points = vec![];
    let mut active = HashSet::new();

    let min = octree.pos();
    let max = min + octree.size();

    let first = rng.sample(Uniform::new(min, max).unwrap());
    points.push(first);
    active.insert(0);
    octree.insert(first);

    while !active.is_empty() {
        let i = *active.iter().choose(rng).unwrap();
        let seed = points[i];
        if let Some(point) = try_sample_point(rng, seed, min, max, &octree) {
            active.insert(points.len());
            points.push(point);
            octree.insert(point);
        } else {
            active.remove(&i);
        }
    }

    points
}
