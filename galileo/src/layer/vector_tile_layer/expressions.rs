use core::f64;
use crate::error::GalileoError;
use crate::Color;



/// Context for the interpolation 
struct InterpolateContext{
    current_color: Color,
    current_resolution: f64,
}

struct Expression<T> {
        start_value: T,
        end_value: T,
        max_resolution:Option<f64>,
        min_resolution:Option<f64>,
        interpolation_type: Interpolation,
        interpolation_args: Vec<i32>,
}

enum Interpolation{
    Linear,
    Exponential,
    Cubic,
}

pub trait Provider<T>{
    fn get_value( &self,context:InterpolateContext)->Result<T,GalileoError>;
}

impl Provider<Color> for Color{ 
    fn get_value(&self,_:InterpolateContext)->Result<Color,GalileoError>{
        Ok(*self)
    }
}

impl Provider<Color> for Expression<Color> {
    fn get_value( &self,context:InterpolateContext)->Result<Color,GalileoError>{ 

        let k: Color = match (self.max_resolution,self.min_resolution) {

                (Some(max_resolution),Some(min_resolution))=>{
                    const EPS:f64 = 10e-6;

                    let resolution_range: f64 = (max_resolution - min_resolution).clamp(EPS,f64::MAX);
                    // individual ratios for each field
                    let kr = (self.end_value.r() - self.start_value.r()) as f64 / resolution_range;
                    let kg = (self.end_value.g() - self.start_value.g()) as f64 / resolution_range;
                    let kb = (self.end_value.b() - self.start_value.b()) as f64 / resolution_range;
                    let ka = (self.end_value.a() - self.start_value.a()) as f64 / resolution_range;

                    let offset = (context.current_resolution - min_resolution).clamp(0.0, resolution_range);

                    Color::rgba(
                            (self.start_value.r() as f64 + kr*offset).clamp(0.0,255.0) as u8,
                            (self.start_value.g() as f64 + kg*offset).clamp(0.0,255.0) as u8,
                            (self.start_value.b() as f64 + kb*offset).clamp(0.0,255.0) as u8,
                            (self.start_value.a() as f64 + ka*offset).clamp(0.0,255.0) as u8,
                    )

                }
                (_,_)=>{
                    Err(GalileoError::Configuration("Unexpectedly missing resolution configurations!".to_string()))?
                }
            };
            Ok(k)
    }
}
