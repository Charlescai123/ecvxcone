# ecvxcone

[![Ubuntu 20.04/22.04](https://img.shields.io/badge/Ubuntu-20.04/22.04-red?logo=ubuntu)](https://ubuntu.com/)
[![ROS2 Foxy/Humble](https://img.shields.io/badge/ros2-foxy/humble-brightgreen.svg?logo=ros)](https://wiki.ros.org/foxy)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg?logo=apache)](https://opensource.org/license/apache-2-0)

---


## 💡 User Guide

### ⚙️ Dependencies

* ROS2 Humble
* Jetpack 6.2
* L4T (Linux for Tegra) r36.4.3

---

ecvxcone (**e**mbedded **CVX** for **cone** programming) is a lightweight solver tailored for embedded system use. It supports general cone programming optimization problems including LP, QP, SCOP and SDP.

> ![NOTE]
> Currently this library supports the constraints in the linear form. The support for quadratic form of constraints may be added in the future version.

Inspired by [cvxopt](https://github.com/cvxopt/cvxopt), `ecvxcone` rewrites the solver logic in pure C to better suit embedded and real-time applications.

We recommend purchasing an NVMe SSD, as it offers significantly faster read and write speeds compared to the built-in 64GB eMMC. Once installed, use the [SDK Manager](https://developer.nvidia.com/sdk-manager) to flash the JetPack SDK directly onto the NVMe drive.

Vision: Despite the growing demand for real-time optimization in embedded systems, there remains a lack of open-source conic solvers designed specifically for deployment on resource-constrained devices. This repository aims to bridge that gap by providing a lightweight, efficient conic solver framework tailored for embedded applications—especially in the context of the rising adoption of machine learning at the edge. 

Moreover, there exists a significant disconnect between academic research and practical industrial deployment. Many optimization algorithms are proposed in theory but lack accessible implementations or validation due to hardware limitations and the absence of suitable open-source tooling. We hope this repository can serve as a foundation for both academic experimentation and real-world deployment, fostering collaboration across research and industry.


| Argument         | Meaning       | Type          |
| :-------------:    | :-------------: | :-------------: |
| `LP` (Linear Programming)      | $$ \min_{\mathbf{x}} ~\mathbf{c}^T\mathbf{x} + d \\ \begin{aligned} \text{s.t.} ~&\mathbf{A}\mathbf{x} \leq \mathbf{b} \\ &\mathbf{G}\mathbf{x} = \mathbf{h} \end{aligned} $$ | String          | 
| `QP` (Quadratic Programming)   | $$ \min_{\mathbf{x}} ~\frac{1}{2}\mathbf{x}^T\mathbf{Q}\mathbf{x} + \mathbf{p}^T\mathbf{x} + r \\ \begin{aligned} \text{s.t.} ~&\mathbf{A}\mathbf{x} \leq \mathbf{b} \\ &\mathbf{G}\mathbf{x} = \mathbf{h} \end{aligned} $$ | String          |
| `SOCP` (Second-Order Conic Programming)    | $$\min_{\mathbf{x}} ~\mathbf{f}^T\mathbf{x} \\ \text{s.t.} ~ \|\|\mathbf{A}_i\mathbf{x} + \mathbf{b}_i\|\|_2 \leq \mathbf{c}_i^T\mathbf{x} + d_i, ~i = 1,...,n \\ \mathbf{G}\mathbf{x} = \mathbf{h} $$  | Dict            |
| `SDP` (Semi-Definite Programming) | $$ \min_{\mathbf{x}} ~\mathbf{c}^T\mathbf{x} + d \\ \begin{aligned} \text{s.t.} ~&\mathbf{A}\mathbf{x} \leq \mathbf{b} \\ &\mathbf{G}\mathbf{x} = \mathbf{h} \end{aligned} $$ | List of Strings |


For remote onboard Jetson debug, you can use other remote connection tool (e.g., *Anydesk*, *Teamviewer*). However, these softwares are not typically useful if the Jetson is not connected to a monitor. According to the [solution](https://askubuntu.com/questions/453109/add-fake-display-when-no-monitor-is-plugged-in), there're two options:

1. purchase a dummy display plug [DP Port](https://www.amazon.com/FUERAN-DP-DisplayPort-emulator-2560x1600/dp/B075PTQ4NH/ref=sr_1_2_sspa?dib=eyJ2IjoiMSJ9._2VBeAusIsfeL9L3oHOyN6PCekAraiSOV3Mbqo9epsJwtAFnpeQKXHkz0wJTW8nPMM9W9X-Z_Sbt1gaplwMw2RqV9BnFry5G4bcKFI4PVCb8FD_8RxNh8B5D97RcYCU7aVqnbDOkPJZwz5UVYRL-jNVwVQtxFcuWtiiWVWm0XvcXtdtGp9HA9WySLN1MFOEmpgGPhkNvBDvJuxqC3YC28CEOz7CkhmC0gsXiwtJNmkA.K023GTNCJhfrzl89n0sznwDJ5qqfZ3KOK0XIoN7h2XE&dib_tag=se&keywords=dp%2Bport%2Bdummy%2Bport&qid=1750013547&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1)
2. turn the video output to a 
```bash
sudo apt-get install xserver-xorg-video-dummy
```

Then add the content in `/etc/X11/xorg.conf`:
```bash
Section "Device"
    Identifier  "Configured Video Device"
    Driver      "dummy"
    # Default is 4MiB, this sets it to 16MiB
    VideoRam    16384
EndSection

Section "Monitor"
    Identifier  "Configured Monitor"
    HorizSync 31.5-48.5
    VertRefresh 50-70
EndSection

Section "Screen"
    Identifier  "Default Screen"
    Monitor     "Configured Monitor"
    Device      "Configured Video Device"
    DefaultDepth 24
    SubSection "Display"
    Depth 24
    Modes "1024x800"
    EndSubSection
EndSection
```

Package Dependency:

* *Coal (Extension of the Flexible Collision Library) - 3.0.1*
* *pinocchio (Rigid Body Dynamics Library) - 3.7.0*
* *Mujoco - 3.2.7*


### 🔨 Setup

1. Clone this repository:

```bash
git clone git@github.com:Charlescai123/edge-learning-go2.git
```

## ⏳ To Do ##

<!-- * [x] Add BEV map to the repo -->
<!-- * [x] Fast Marching Method (FMM) implementation -->

* [ ] Code Refactorization
* [ ] Incorporate more challenging scenarios
    * [x] Dense forests (sandy terrain, trees)
    * [ ] inclined staircases, and rainy conditions
* [ ] Go2 real robot deployment
    * [ ] Gazebo real-time testing
    * [ ] ROS/ROS2 integration
* [ ] Restructure the code as FSM and add teleoperation (optional)
* [ ] Migration to Isaac-Lab

---

## 🏷️ Misc

---

asdsa

## 📝 References


[2] Lieven Vandenberghe. Conic Programming. Department of Electrical and Computer Engineering, UCLA. Available at: https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

[3] Martin S. Andersen and Lieven Vandenberghe. Introduction to Mathematical Optimization. Unpublished manuscript. Available at: https://www.seas.ucla.edu/~vandenbe/publications/mlbook.pdf


---

## 🎉 Acknowledgments

- [cvxopt](https://github.com/cvxopt/cvxopt): Base references for implementation of cone programming.
- [cvxpygen](https://github.com/cvxgrp/cvxpygen): Some base codes for modeling DPP-compliant problem.