//! Module that provides support for step and interpolate expressions for Color and Number types
use std::cmp::Ordering;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Color;

/// Wrapper over arguments for Interpolation Functions
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum InterpolationArgs<T> {
    /// Linear variant of interpolation arguments
    Linear(LinearInterpolationArgs<T>),
    /// Exponential variant of interpolation arguments
    Exponential(ExponentialInterpolationArgs<T>),
}

impl<T: Copy> InterpolationArgs<T> {
    fn step_values(&self) -> &BTreeSet<StepValue<T>> {
        match self {
            Self::Linear(args) => &args.step_values,
            Self::Exponential(args) => &args.step_values,
        }
    }
}
/// Arguments for Linear Interpolation Function
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinearInterpolationArgs<T> {
    step_values: BTreeSet<StepValue<T>>,
}

impl<T: Copy> LinearInterpolationArgs<T> {
    /// Returns a new instance of `LinearInterpolationArgs`
    pub fn new(step_values: Vec<StepValue<T>>) -> Result<Self, String> {
        if step_values.len() < 2 {
            return Err("At least 2 step values required".to_string());
        }
        Ok(Self {
            step_values: step_values.into_iter().collect::<BTreeSet<_>>(),
        })
    }
}

/// Arguments for Exponential Interpolation Function
#[derive(Clone, Debug, Serialize, PartialEq, Deserialize)]
pub struct ExponentialInterpolationArgs<T> {
    base: i32,
    step_values: BTreeSet<StepValue<T>>,
}

impl<T: Copy> ExponentialInterpolationArgs<T> {
    /// Returns a new instance of `ExponentialInterpolationArgs`
    pub fn new(base: i32, step_values: Vec<StepValue<T>>) -> Result<Self, String> {
        if base <= 0 {
            return Err("Base must be positive".to_string());
        }

        if step_values.len() < 2 {
            return Err("At least 2 step values required".to_string());
        }
        Ok(Self {
            base,
            step_values: step_values.into_iter().collect::<BTreeSet<_>>(),
        })
    }
}

/// Wrapper type for each step value
/// i.e. resolution and a Color or a Number
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct StepValue<T> {
    /// Minimum resolution for the step to be used.
    pub resolution: f64,
    /// Literal value for the step..
    pub step_value: T,
}

impl<T> PartialEq for StepValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.resolution == other.resolution
    }
}

impl<T> Eq for StepValue<T> {}

impl<T> PartialOrd for StepValue<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for StepValue<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.resolution.total_cmp(&other.resolution)
    }
}

/// Type used to define expressions for interpolation
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct InterpolateExpression<T> {
    interpolation_args: InterpolationArgs<T>,
}
/// Type used to define Step Function
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct StepExpression<T> {
    /// Each stop value maps the resolution to the T type
    /// If, the current resolution is greater than step resolution
    /// the value T maps to the T value where the step resolution is
    /// less than that of current resolution.
    step_values: BTreeSet<StepValue<T>>,
    default_value: T,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ResolutionValueRange<T> {
    max_resolution: f64,
    min_resolution: f64,
    start_value: T,
    end_value: T,
}

/// StyleValue introduces simple, interpolation and step functions to be used as features that are
/// evaluated to give color values on the basis of the zoom level
#[derive(Clone, Debug, Serialize, PartialEq)]
pub enum StyleValue<T> {
    /// Style variant to wrap interpolate function
    Interpolate(InterpolateExpression<T>),
    /// Style variant to wrap step function
    Steps(StepExpression<T>),
    /// Style variant simple values(i.e. like Color::BLACK)
    Simple(T),
}

impl<'de, T> Deserialize<'de> for StyleValue<T>
where
    T: for<'a> Deserialize<'a> + Copy,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde_json::Value;

        let value = Value::deserialize(deserializer)?;

        match value {
            Value::Array(arr) => match arr.first().and_then(|v| v.as_str()) {
                Some("interpolate") => {
                    let expr =
                        InterpolateExpression::from_array(arr).map_err(serde::de::Error::custom)?;
                    Ok(StyleValue::Interpolate(expr))
                }
                Some("step") => {
                    let expr = StepExpression::from_array(arr).map_err(serde::de::Error::custom)?;
                    Ok(StyleValue::Steps(expr))
                }
                _ => Err(serde::de::Error::custom("Unknown expression")),
            },
            other => {
                let literal =
                    serde_json::from_value::<T>(other).map_err(serde::de::Error::custom)?;
                Ok(StyleValue::Simple(literal))
            }
        }
    }
}

impl<T> InterpolateExpression<T>
where
    T: Copy + for<'de> Deserialize<'de>,
{
    fn from_array(arr: Vec<serde_json::Value>) -> Result<Self, String> {
        if arr.len() < 6 {
            return Err("Invalid interpolate expression".into());
        }
        let interpolation_args = match &arr[1] {
            serde_json::Value::Array(v) if v[0] == "linear" => {
                InterpolationArgs::Linear(LinearInterpolationArgs::new(parse_step(&arr[3..])?)?)
            }
            serde_json::Value::Array(v) if v[0] == "exponential" => {
                let base = v.get(1).and_then(|b| b.as_i64()).unwrap_or(1) as i32;
                InterpolationArgs::Exponential(ExponentialInterpolationArgs::new(
                    base,
                    parse_step(&arr[3..])?,
                )?)
            }
            _ => return Err("Unsupported interpolation type".into()),
        };
        Ok(Self { interpolation_args })
    }
}

impl<T> StepExpression<T>
where
    T: Copy + for<'de> Deserialize<'de>,
{
    fn from_array(arr: Vec<serde_json::Value>) -> Result<Self, String> {
        if arr.len() < 4 {
            return Err("Invalid step expression".into());
        }
        let default_value = T::deserialize(arr[2].clone()).map_err(|_| "Invalid default value")?;
        let step_values = parse_step(&arr[3..])?;
        Self::new(default_value, step_values)
    }
}

fn parse_step<T>(raw: &[serde_json::Value]) -> Result<Vec<StepValue<T>>, String>
where
    T: for<'de> Deserialize<'de>,
{
    if !raw.len().is_multiple_of(2) {
        return Err("Stops must be pairs".into());
    }

    let mut out = Vec::new();

    for pair in raw.chunks(2) {
        let zoom_level = pair[0].as_i64().ok_or("Invalid stop resolution")?;

        let resolution = 156543.03392800014 / 2.0f64.powi(zoom_level as i32);

        let value = T::deserialize(pair[1].clone()).map_err(|_| "Invalid stop value")?;

        out.push(StepValue {
            resolution,
            step_value: value,
        });
    }

    Ok(out)
}

impl From<Color> for StyleValue<Color> {
    fn from(color_val: Color) -> Self {
        Self::Simple(color_val)
    }
}

impl From<f64> for StyleValue<f64> {
    fn from(num: f64) -> Self {
        Self::Simple(num)
    }
}

impl StyleValue<Color> {
    /// Evaluates value of Color depending upon the type of expression used.
    pub fn get_value(&self, current_resolution: f64) -> Color {
        match self {
            StyleValue::Simple(t) => *t,
            StyleValue::Interpolate(expression) => expression.evaluate(current_resolution),
            StyleValue::Steps(expression) => expression.evaluate(current_resolution),
        }
    }
}
impl StyleValue<f64> {
    /// Evaluates value of Number depending upon the type of expression used.
    pub fn get_value(&self, current_resolution: f64) -> f64 {
        match self {
            StyleValue::Simple(t) => *t,
            StyleValue::Interpolate(expression) => expression.evaluate(current_resolution),
            StyleValue::Steps(expression) => expression.evaluate(current_resolution),
        }
    }
}

impl<T> InterpolateExpression<T> {
    /// Returns a new instance of `InterpolateExpression`
    pub fn new(interpolation_args: InterpolationArgs<T>) -> Self {
        Self { interpolation_args }
    }
    fn get_boundary_value(&self, current_resolution: f64) -> T
    where
        T: Copy,
    {
        let step_values = self
            .interpolation_args
            .step_values()
            .iter()
            .collect::<Vec<_>>();

        if current_resolution < step_values[0].resolution {
            step_values[0].step_value
        } else {
            step_values[step_values.len() - 1].step_value
        }
    }
}

impl InterpolateExpression<Color> {
    /// Evaluates Color value of the expression on the basis of the build
    fn evaluate(&self, current_resolution: f64) -> Color {
        if let Some(resolution_value_range) = get_resolution_value_range(self, current_resolution) {
            self.interpolate_color(&resolution_value_range, current_resolution)
        } else {
            self.get_boundary_value(current_resolution)
        }
    }
}

impl InterpolateExpression<f64> {
    /// Evaluates Numeric value of the expression on the basis of the build
    fn evaluate(&self, current_resolution: f64) -> f64 {
        if let Some(resolution_value_range) = get_resolution_value_range(self, current_resolution) {
            self.interpolate_number(&resolution_value_range, current_resolution)
        } else {
            self.get_boundary_value(current_resolution)
        }
    }
}

enum Channel {
    R,
    G,
    B,
    A,
}

impl InterpolateExpression<Color> {
    fn interpolate_color(
        &self,
        rv_range: &ResolutionValueRange<Color>,
        current_resolution: f64,
    ) -> Color {
        let channels = [Channel::R, Channel::G, Channel::B, Channel::A];

        let values: Vec<u8> = channels
            .iter()
            .map(|ch| self.interpolate_color_channel(current_resolution, rv_range, ch))
            .collect::<Vec<_>>();

        Color::rgba(values[0], values[1], values[2], values[3])
    }

    fn interpolate_color_channel(
        &self,
        current_resolution: f64,
        rv_range: &ResolutionValueRange<Color>,
        channel: &Channel,
    ) -> u8 {
        let (start, end) = match channel {
            Channel::R => (rv_range.start_value.r(), rv_range.end_value.r()),
            Channel::G => (rv_range.start_value.g(), rv_range.end_value.g()),
            Channel::B => (rv_range.start_value.b(), rv_range.end_value.b()),
            Channel::A => (rv_range.start_value.a(), rv_range.end_value.a()),
        };
        match &self.interpolation_args {
            InterpolationArgs::Linear(_) => linear_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                start as f64,
                end as f64,
                current_resolution,
            ) as u8,
            InterpolationArgs::Exponential(args) => exponential_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                start as f64,
                end as f64,
                current_resolution,
                args.base,
            ) as u8,
        }
    }
}

impl InterpolateExpression<f64> {
    fn interpolate_number(
        &self,
        rv_range: &ResolutionValueRange<f64>,
        current_resolution: f64,
    ) -> f64 {
        match &self.interpolation_args {
            InterpolationArgs::Linear(_) => linear_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value,
                rv_range.end_value,
                current_resolution,
            ),
            InterpolationArgs::Exponential(args) => exponential_interpolation(
                rv_range.min_resolution,
                rv_range.max_resolution,
                rv_range.start_value,
                rv_range.end_value,
                current_resolution,
                args.base,
            ),
        }
    }
}

fn exponential_interpolation(
    x_start: f64,
    x_end: f64,
    y_start: f64,
    y_end: f64,
    x0: f64,
    base: i32,
) -> f64 {
    let t: f64 = ((x0 - x_start) / (x_end - x_start).clamp(f64::EPSILON, f64::MAX)).clamp(0.0, 1.0);

    let base: f64 = base as f64;
    let t = if (base - 1.0).abs() > f64::EPSILON {
        (base.powf(t) - 1.0) / (base - 1.0)
    } else {
        t
    };

    let offset = y_end - y_start;

    (y_start + t * (offset)).clamp(0.0, 255.0)
}

fn linear_interpolation(x_start: f64, x_end: f64, y_start: f64, y_end: f64, x0: f64) -> f64 {
    let x_range: f64 = (x_end - x_start).clamp(f64::EPSILON, f64::MAX);

    let k = (y_end - y_start) / x_range;

    let offset = (x0 - x_start).clamp(0.0, x_range);
    y_start + k * offset
}

fn get_resolution_value_range<T: Copy>(
    expression: &InterpolateExpression<T>,
    current_resolution: f64,
) -> Option<ResolutionValueRange<T>> {
    let step_values: &BTreeSet<StepValue<T>> = expression.interpolation_args.step_values();
    // Try to find a matching window in step_values
    step_values
        .iter()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| current_resolution >= w[0].resolution && current_resolution <= w[1].resolution)
        .map(|w| ResolutionValueRange {
            min_resolution: w[0].resolution,
            max_resolution: w[1].resolution,
            start_value: w[0].step_value,
            end_value: w[1].step_value,
        })
}

impl<T: Copy> StepExpression<T> {
    /// Returns a new instance of `StepExpression`
    pub fn new(default_value: T, step_values: Vec<StepValue<T>>) -> Result<Self, String> {
        if step_values.is_empty() {
            return Err("At least 1 step value required".to_string());
        }
        Ok(Self {
            default_value,
            step_values: step_values.into_iter().collect::<BTreeSet<_>>(),
        })
    }
}
impl<T: Copy> StepExpression<T> {
    /// Evaluates generic expression by giving stepwise value
    /// of color on basis of zoom
    fn evaluate(&self, current_resolution: f64) -> T {
        if let Some(w) = self
            .step_values
            .iter()
            .collect::<Vec<_>>()
            .windows(2)
            .find(|w| {
                current_resolution >= w[0].resolution && current_resolution <= w[1].resolution
            })
        {
            w[0].step_value
        } else if current_resolution
            < self
                .step_values
                .iter()
                .nth(0)
                .expect("value at 0th position")
                .resolution
        {
            self.default_value
        } else {
            self.step_values
                .iter()
                .nth(self.step_values.len() - 1)
                .expect("value at end position")
                .step_value
        }
    }
}

#[cfg(test)]
mod number_tests {
    use super::*;

    #[test]
    fn test_get_resolution_value_range_out_of_bounds_f64() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 25.0,
                step_value: 0.0,
            },
            StepValue {
                resolution: 50.0,
                step_value: 100.0,
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression {
            interpolation_args: InterpolationArgs::Linear(args),
        };

        assert!(get_resolution_value_range(&expr, 75.0).is_none());
        assert!(get_resolution_value_range(&expr, 20.0).is_none());
    }

    #[test]
    fn linear_interpolation_bounds_f64() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 25.0,
                step_value: 0.0,
            },
            StepValue {
                resolution: 50.0,
                step_value: 100.0,
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Linear(args));

        assert_eq!(expr.evaluate(20.0).round(), 0.0);
        assert_eq!(expr.evaluate(150.0).round(), 100.0);
    }

    #[test]
    fn linear_interpolation_unordered_f64() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 50.0,
                step_value: 100.0,
            },
            StepValue {
                resolution: 0.0,
                step_value: 0.0,
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Linear(args));

        assert_eq!(expr.evaluate(25.0).round(), 50.0);
    }

    #[test]
    fn linear_interpolation_f64() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 0.0,
                step_value: 0.0,
            },
            StepValue {
                resolution: 50.0,
                step_value: 100.0,
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Linear(args));

        assert_eq!(expr.evaluate(25.0).round(), 50.0);
    }

    #[test]
    fn exponential_bounds_f64() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 10.0,
                    step_value: 0.0,
                },
                StepValue {
                    resolution: 50.0,
                    step_value: 100.0,
                },
            ],
        )
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Exponential(args));

        assert_eq!(expr.evaluate(5.0).round(), 0.0);
        assert_eq!(expr.evaluate(150.0).round(), 100.0);
    }

    #[test]
    fn exponential_interpolation_unordered_f64() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 50.0,
                    step_value: 100.0,
                },
                StepValue {
                    resolution: 0.0,
                    step_value: 0.0,
                },
                StepValue {
                    resolution: 75.0,
                    step_value: 200.0,
                },
            ],
        )
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Exponential(args));

        assert_eq!(expr.evaluate(25.0).round(), 41.0);
        assert_eq!(expr.evaluate(60.0).round(), 132.0);
    }

    #[test]
    fn exponential_interpolation_f64() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 0.0,
                    step_value: 0.0,
                },
                StepValue {
                    resolution: 50.0,
                    step_value: 100.0,
                },
                StepValue {
                    resolution: 75.0,
                    step_value: 200.0,
                },
            ],
        )
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Exponential(args));

        assert_eq!(expr.evaluate(25.0).round(), 41.0);
        assert_eq!(expr.evaluate(60.0).round(), 132.0);
    }

    #[test]
    fn test_step_expression_bounds_f64() {
        let expr = StepExpression::<f64>::new(
            10.0,
            vec![
                StepValue {
                    resolution: 10.0,
                    step_value: 20.0,
                },
                StepValue {
                    resolution: 20.0,
                    step_value: 30.0,
                },
            ],
        )
        .expect("failed to create step expression");

        assert_eq!(expr.evaluate(5.0).round(), 10.0);
        assert_eq!(expr.evaluate(30.0).round(), 30.0);
    }

    #[test]
    fn test_step_expression_f64() {
        let expr = StepExpression::<f64>::new(
            10.0,
            vec![
                StepValue {
                    resolution: 10.0,
                    step_value: 20.0,
                },
                StepValue {
                    resolution: 20.0,
                    step_value: 30.0,
                },
                StepValue {
                    resolution: 30.0,
                    step_value: 40.0,
                },
            ],
        )
        .expect("failed to create step expression");

        assert_eq!(expr.evaluate(15.0).round(), 20.0);
        assert_eq!(expr.evaluate(25.0).round(), 30.0);
    }
}

#[cfg(test)]
mod color_tests {
    use super::*;

    #[test]
    fn test_get_resolution_value_range_out_of_bounds() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 25.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression {
            interpolation_args: InterpolationArgs::Linear(args),
        };

        assert!(get_resolution_value_range(&expr, 75.0).is_none());
        assert!(get_resolution_value_range(&expr, 20.0).is_none());
    }

    #[test]
    fn linear_interpolation_bounds() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 25.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Linear(args));

        assert_eq!(expr.evaluate(20.0), Color::rgba(0, 0, 0, 0));
        assert_eq!(expr.evaluate(150.0), Color::rgba(128, 128, 128, 128));
    }

    #[test]
    fn linear_interpolation_unordered() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
            StepValue {
                resolution: 0.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Linear(args));

        assert_eq!(expr.evaluate(25.0), Color::rgba(64, 64, 64, 64));
    }

    #[test]
    fn linear_interpolation() {
        let args = LinearInterpolationArgs::new(vec![
            StepValue {
                resolution: 0.0,
                step_value: Color::rgba(0, 0, 0, 0),
            },
            StepValue {
                resolution: 50.0,
                step_value: Color::rgba(128, 128, 128, 128),
            },
        ])
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Linear(args));

        assert_eq!(expr.evaluate(25.0), Color::rgba(64, 64, 64, 64));
    }

    #[test]
    fn exponential_bounds() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 10.0,
                    step_value: Color::rgba(0, 0, 0, 0),
                },
                StepValue {
                    resolution: 50.0,
                    step_value: Color::rgba(128, 128, 128, 128),
                },
            ],
        )
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Exponential(args));

        assert_eq!(expr.evaluate(5.0), Color::rgba(0, 0, 0, 0));
        assert_eq!(expr.evaluate(150.0), Color::rgba(128, 128, 128, 128));
    }

    #[test]
    fn exponential_interpolation_unordered() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 50.0,
                    step_value: Color::rgba(128, 128, 128, 128),
                },
                StepValue {
                    resolution: 0.0,
                    step_value: Color::rgba(0, 0, 0, 0),
                },
                StepValue {
                    resolution: 75.0,
                    step_value: Color::rgba(200, 200, 200, 200),
                },
            ],
        )
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Exponential(args));

        assert_eq!(expr.evaluate(25.0), Color::rgba(53, 53, 53, 53));
        assert_eq!(expr.evaluate(60.0), Color::rgba(151, 151, 151, 151));
    }

    #[test]
    fn exponential_interpolation() {
        let args = ExponentialInterpolationArgs::new(
            2,
            vec![
                StepValue {
                    resolution: 0.0,
                    step_value: Color::rgba(0, 0, 0, 0),
                },
                StepValue {
                    resolution: 50.0,
                    step_value: Color::rgba(128, 128, 128, 128),
                },
                StepValue {
                    resolution: 75.0,
                    step_value: Color::rgba(200, 200, 200, 200),
                },
            ],
        )
        .expect("failed to create interpolation arguments");

        let expr = InterpolateExpression::new(InterpolationArgs::Exponential(args));

        assert_eq!(expr.evaluate(25.0), Color::rgba(53, 53, 53, 53));
        assert_eq!(expr.evaluate(60.0), Color::rgba(151, 151, 151, 151));
    }

    #[test]
    fn test_step_expression_bounds() {
        let expr = StepExpression::<Color>::new(
            Color::from_hex("#f0f0f0"),
            vec![
                StepValue::<Color> {
                    resolution: 10.0,
                    step_value: Color::from_hex("#fafafa"),
                },
                StepValue::<Color> {
                    resolution: 20.0,
                    step_value: Color::from_hex("#1d1d1d"),
                },
            ],
        )
        .expect("failed to create step expression");

        assert_eq!(expr.evaluate(5.0), Color::from_hex("#f0f0f0"));
        assert_eq!(expr.evaluate(30.0), Color::from_hex("#1d1d1d"));
    }

    #[test]
    fn test_step_expression() {
        let expr = StepExpression::<Color>::new(
            Color::from_hex("#f0f0f0"),
            vec![
                StepValue::<Color> {
                    resolution: 10.0,
                    step_value: Color::from_hex("#fafafa"),
                },
                StepValue::<Color> {
                    resolution: 20.0,
                    step_value: Color::from_hex("#1d1d1d"),
                },
                StepValue::<Color> {
                    resolution: 30.0,
                    step_value: Color::from_hex("#1a1a1a"),
                },
            ],
        )
        .expect("failed to create step expression");

        assert_eq!(expr.evaluate(15.0), Color::from_hex("#fafafa"));
        assert_eq!(expr.evaluate(25.0), Color::from_hex("#1d1d1d"));
    }
}
