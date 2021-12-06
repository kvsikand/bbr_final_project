# BBR Final Project Simulator
This simulator is a part of the final project for Brain, Body, and Robotics.

Here we simulate the stiffness ellipse for a three-jointed arm-like robot with a tendon-based articulation structure.

The aim of this simulator is to XXXX?

We support simulating the endpoint stiffness ellipse given arbitrary limb lengths and tendon stiffness parameters. We also support both bi-articular and mono-articular tendons, to show how these can affect endpoint stiffnesss.

In mono-articular mode, there are 3 simulated tendons, each of which actuates a single joint.
In bi-articular mode, there are 5 simulated tendons, 2 of which are bi-articular.

### Running the Simulator

See the files run_simulator_*.sh for examples of running the code in mono-articular and bi-articular modes.

In mono-articular mode, the order of tendon stiffnesses & tendon lengths is:
 - shoulder
 - elbow
 - hand

In bi-articular mode, the order of tendon stiffnesses is:
 - shoulder
 - shoulder+elbow
 - elbow
 - elbow+hand
 - hand

And the order of tendon lengths is:
 - shoulder
 - elbow
 - hand
 - shoulder+elbow
 - elbow+hand

The bi-articular tendon lengths are assumed to be equidistant from both joints that they affect


This project requires python, numpy, and opencv