# FMCW Radar Design for Target Detection and Velocity Estimation

## 🛰️ Project Overview
This project implements a complete **77 GHz FMCW Radar** simulation in MATLAB. It is designed to detect target range and velocity using advanced signal processing techniques, including 2D FFT and CA-CFAR.

## 📁 Repository Structure
* `/src`: Contains the MATLAB source code (`main_radar_project.m`, `radar_config.m`).
* `/docs`: Contains the detailed technical project report.

## 🛠️ System Specifications
Based on the simulation parameters:
- **Carrier Frequency:** 76.5 GHz
- **Bandwidth:** 1 GHz (Providing high range resolution)
- **Max Range:** 250 m
- **Chirp Duration:** 8 μs

## 🔍 Signal Processing Pipeline
1. **Signal Generation:** Mixing Transmit (TX) and Receive (RX) signals to produce the Beat Signal.
2. **Range FFT:** 1D FFT to resolve the distance of targets.
3. **Doppler FFT:** Processing across chirps to calculate radial velocity.
4. **CA-CFAR:** A Cell-Averaging Constant False Alarm Rate algorithm to filter noise and isolate targets.



## 🚀 How to Run
1. Clone the repository: `git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git`
2. Open MATLAB and navigate to the `src` folder.
3. Open `main_radar_project.m`.
4. Change the `scenario_id` in line 7 to test different cases (e.g., `1` for Normal, `2` for Low SNR, `9` for Multiple Objects).
5. Run the script.

## 📊 Results
The system successfully identifies targets even in low SNR conditions. For example, in Scenario 1 (Normal Case), the radar detects objects at 100m and 220m with high accuracy.

## 👥 Authors
* Mohaned Waleed Hosny
* Muhammad Sameer Abdelhamid
