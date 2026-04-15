use glam::Vec3;
use rand::{SeedableRng, rngs::SmallRng};

use crate::{app::PortalPair, poisson::octree::Octree};

mod octree;
mod sphere_sampling;

#[derive(Debug)]
pub struct Simulation {
    points: Vec<Vec3>,
}
impl Simulation {
    pub fn new(pos: Vec3, size: f32, _portals: &[PortalPair]) -> Self {
        let octree = Octree::new(pos, size, size.log2() as usize);

        let points = sphere_sampling::sample_points(&mut SmallRng::from_os_rng(), octree);

        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                if points[i].distance_squared(points[j]) < 1.0 {
                    panic!("{}, {}", points[i], points[j]);
                }
            }
        }

        Self { points }
    }
}
