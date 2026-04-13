use std::collections::HashSet;
use std::f32::consts::TAU;

use glam::Vec3;
use rand::distr::{Distribution, Uniform};
use rand::{Rng, seq::IteratorRandom};

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
        Self { center: Vec3::ZERO, r: 1.0 }
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

fn check_constraints(point: Vec3, r: f32, min: Vec3, max: Vec3, existing: &[Vec3]) -> bool {
    if !((min.x..max.x).contains(&point.x) && (min.y..max.y).contains(&point.y) && (min.z..max.z).contains(&point.z)) {
        return false;
    }
    for p in existing {
        if point.distance_squared(*p) < r.powi(2) {
            return false;
        }
    }
    true
}
fn try_sample_point(rng: &mut impl Rng, seed: Vec3, min: Vec3, max: Vec3, existing: &[Vec3]) -> Option<Vec3> {
    for _ in 0..10 {
        let r = 1.0;
        let new = rng.sample(SphereShellDistribution::new(seed, r));
        if check_constraints(new, r, min, max, existing) {
            return Some(new);
        }
    }
    None
}
pub fn sample_points(rng: &mut impl Rng, min: Vec3, max: Vec3) -> Vec<Vec3> {
    let mut points = vec![];
    let mut active = HashSet::new();

    points.push(rng.sample(Uniform::new(min, max).unwrap()));
    active.insert(0);
    
    while !active.is_empty() {
        let i = *active.iter().choose(rng).unwrap();
        let seed = points[i];
        if let Some(point) = try_sample_point(rng, seed, min, max, &points) {
            active.insert(points.len());
            points.push(point);
        } else {
            active.remove(&i);
        }
    }
    println!("len: {}", points.len());

    points
}
