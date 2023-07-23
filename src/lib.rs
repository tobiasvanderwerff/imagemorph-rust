use std::process;
use std::ops::{Div, Mul};
use std::error::Error;
use rand::Rng;
use image::io::Reader as ImageReader;
use image::{Rgb, RgbImage};
use ndarray::Array2;


pub fn run(cfg: &Config) -> Result<(), Box<dyn Error>> {
    // Load the image
    let img = load_image_rgb(&cfg.filename)?;
    let (img_h, img_w) = (img.height() as usize, img.width() as usize);
    dbg!(img_h, img_w);

    let img_res = rubbersheet(img, img_h, img_w, cfg.amp, cfg.sigma)?;

    // Write to disk
    let outpath = String::from("out.png");
    img_res.save(&outpath)?;  // return error if fails

    println!("Output written to {outpath}");

    Ok(())
}

fn rubbersheet(img: RgbImage, h: usize, w: usize, amp: f32, sigma: f32) -> Result<RgbImage, Box<dyn Error>> {
    let mut res = img.clone();

    // Check that 'sigma' is not too large
    if sigma > h as f32 / 2.5 || sigma > w as f32 / 2.5 {
		eprintln!("- Warning: Gaussian smoothing kernel too large for the input image.");
    }

	// Check that 'sigma' is not negative
	if sigma < 1E-5 {
		eprintln!("- Warning: Gaussian smoothing kernel with negative/zero spread.");
	}

	// Allocate the displacement fields
    let mut d_x = Array2::<f32>::zeros((h, w));
    let mut d_y = Array2::<f32>::zeros((h, w));
    // let mut d_x = [[0; w]; h];
    // let mut d_y = [[0; w]; h];

    compute_displacement_field(&mut d_x, &mut d_y, h, w, amp, sigma);

	// Apply the displacement fields
	apply_displacement_field(&img, &mut res, h, w, &d_x, &d_y);

    Ok(res)
}

fn compute_displacement_field(d_x: &mut Array2<f32>, d_y: &mut Array2<f32>, h: usize, w: usize, amp: f32, sigma: f32) {
	// Allocate the auxiliary displacement fields
    let mut da_x = Array2::<f32>::zeros((h, w));
    let mut da_y = Array2::<f32>::zeros((h, w));
    
	// Allocate and prepare the gaussian smoothing kernel
    let kws = (2.0 * sigma).floor() as usize;
    let mut ker = vec![0_f32; kws+1];
    for (i, el) in ker.iter_mut().enumerate() {
        *el = ((i*i) as f32).mul(-1.0).div(sigma*sigma).exp();
    }

	// Generate the initial random displacement field
    assert!(d_x.dim() == (h, w));
    assert!(d_x.dim() == d_y.dim());
    let mut rng = rand::thread_rng();
    for i in 0..h {
        for j in 0..w {
            d_x[[i, j]] = -1.0 + 2.0 * rng.gen::<f32>();
            d_y[[i, j]] = -1.0 + 2.0 * rng.gen::<f32>();
        }
    }

	// Smooth the random displacement field using the gaussian kernel
    let mut sum_x = 0_f32;
    let mut sum_y = 0_f32;
    let kws_i = kws as i32;
    for i in 0..h {
        for j in 0..w {
            sum_x = 0.0;
            sum_y = 0.0;
            for k in -kws_i..kws_i+1 {
                let mut v = j as i32 + k;
                if v < 0 { v = -v; }
                if v >= w as i32 { v = 2 * (w as i32) - v - 1; }
                let v = v as usize;

                sum_x += d_x[[i, v]] * ker[k.abs() as usize];
                sum_y += d_y[[i, v]] * ker[k.abs() as usize];
            }
            da_x[[i, j]] = sum_x;
            da_y[[i, j]] = sum_y;
        }
    }
    // L190
    for j in 0..w {
        for i in 0..h {
            sum_x = 0.0;
            sum_y = 0.0;
            for k in -kws_i..kws_i+1 {
                let mut u = i as i32 + k;
                if u < 0 { u = -u; }
                if u >= h as i32 { u = 2 * (h as i32) - u - 1; }
                let u = u as usize;

                sum_x += da_x[[u, j]] * ker[k.abs() as usize];
                sum_y += da_y[[u, j]] * ker[k.abs() as usize];
            }
			d_x[[i, j]] = sum_x;
			d_y[[i, j]] = sum_y;
        }
    }

	// Normalize the field
    let mut avg = 0_f32;
    for i in 0..h {
        for j in 0..w {
            avg += ((d_x[[i, j]].powf(2.0) + d_y[[i, j]].powf(2.0))).sqrt();
        }
    }
    avg /= (h * w) as f32;
    for i in 0..h {
        for j in 0..w {
            d_x[[i, j]] = amp * d_x[[i, j]] / avg;
            d_y[[i, j]] = amp * d_y[[i, j]] / avg;
        }
    }
}

// Applies the displacement field to an image: bilinear interpolation is used
fn apply_displacement_field(input: &RgbImage, output: &mut RgbImage, h: usize, w: usize, d_x: &Array2<f32>, d_y: &Array2<f32>) {
	// Bilinear interpolation
    for i in 0..h {
        for j in 0..w {
            let p1: f32 = i as f32 + d_y[[i, j]];
            let p2: f32 = j as f32 + d_x[[i, j]];
            let u0: i32 = p1.floor() as i32;
            let v0: i32 = p2.floor() as i32;
            let f1: f32 = p1 - u0 as f32;
            let f2: f32 = p2 - v0 as f32;

            let mut sumr = 0.0;
            let mut sumg = 0.0;
            let mut sumb = 0.0;
            for idx in 0..4 {
                let (mut u, mut v, f) = match idx {
                    0 => (u0, v0, (1.0 - f1) * (1.0 - f2)),
                    1 => (u0 + 1, v0, f1 * (1.0 - f2)),
                    2 => (u0, v0 + 1, (1.0 - f1) * f2),
                    _ => (u0 + 1, v0 + 1, f1 * f2),
                };
                if u < 0 { u = 0; }
                if u >= h as i32 { u = (h - 1) as i32; }
                if v < 0 { v = 0; }
                if v >= w as i32 { v = (w - 1) as i32; }

                let (r, g, b) = input.get_pixel(v as u32, u as u32).0.into();
                sumr += f * r as f32;
                sumg += f * g as f32;
                sumb += f * b as f32;
            }
            output.put_pixel(j as u32, i as u32, Rgb([sumr as u8, sumg as u8, sumb as u8]));
        }
    }
}

pub struct Config {
    pub filename: String,
    pub amp: f32,
    pub sigma: f32,
}

impl Config {
    pub fn build(args: &[String]) -> Result<Config, &'static str> {
        dbg!(&args);
        if args.contains(&String::from("-h")) { 
            println!("Usage: {} [FILENAME] [AMP] [SIGMA]", args[0]);
            process::exit(0);
        }

        if args.len() < 2 {
            return Err("Not enough arguments");
        }

        let filename = if args.len() > 1 { String::from(&args[1]) } else { String::from("sample-input.png") };
        let amp: f32 = if args.len() > 2 { args[2].parse().expect("Failed to parse 'amp'") } else { 2.5 };
        let sigma: f32 = if args.len() > 3 { args[3].parse().expect("Failed to parse 'sigma'") } else { 9.0 };

        Ok(Config { filename, amp, sigma })
    }
}


fn load_image_rgb(filepath: &str) -> Result<RgbImage, Box<dyn Error>> {
    let img = ImageReader::open(filepath)?.decode()?.into_rgb8();
    Ok(img)
} 

