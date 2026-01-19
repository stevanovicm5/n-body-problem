# Grade: 6
# N-Body Problem – Advanced Programming Techniques Project

## Problem Description

The N-body problem is a classical problem in physics that involves predicting the motion of N bodies that interact with each other through gravitational forces.

Each body exerts a gravitational force on every other body. When the number of bodies is greater than two, the system becomes too complex to solve analytically.  
For this reason, numerical methods are used to approximate the motion of the bodies over time.

The goal of this project is to create a simple numerical simulation of the N-body problem using the Python programming language.

## Project Goal

The main goal of this project is:
- To understand the N-body problem
- To apply basic laws of physics in a programming context
- To use numerical methods to simulate motion
- To practice Python programming and algorithmic thinking

This project focuses on correctness and clarity rather than performance or advanced visualization.

## Planned Solution Approach

The simulation will be based on Newton’s law of universal gravitation and simple numerical integration methods.

### Physical Model

For every pair of bodies, the gravitational force will be calculated using the formula:

`F = G * (m1 * m2) / r²`

where:
- G is the gravitational constant
- m1 and m2 are the masses of the bodies
- r is the distance between the bodies

The total force acting on a body will be the sum of forces from all other bodies.

### Numerical Method

The simulation will use a discrete time step approach.

For each time step:
1. Calculate gravitational forces between all bodies.
2. Compute acceleration using Newton’s second law:  
   `a = F / m`
3. Update velocity:  
   `v = v + a * dt`
4. Update position:  
   `x = x + v * dt`

The Euler method will be used due to its simplicity and suitability for educational purposes.

## Data Representation

Each body will be represented by:
- Mass
- Position (x, y)
- Velocity (vx, vy)

Bodies will be stored in a list or similar data structure.

## Planned Features

- Simulation of an arbitrary number of bodies (N)
- User-defined initial conditions (mass, position, velocity)
- Step-by-step numerical simulation
- Console output of positions and velocities
- Clear and readable Python code

## Technologies to Be Used

- Python 3
- Standard Python libraries (`math`, `random`)

No graphical interface is required for this project.

## Expected Outcome

The expected result of the project is a working Python program that:
- Simulates the motion of multiple bodies under gravity
- Produces reasonable numerical results
- Demonstrates understanding of physics and numerical simulation concepts

## Author

Milan Stevanovic
