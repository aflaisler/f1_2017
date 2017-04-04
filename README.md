## 2017 Formula one driver's performance estimator

This project aims to predict the fastest and average laps of a formula one
driver for the season 2017.


# Motivation

Formula one is by nature a data-driven sports. Using the latest technologies,
engineers are streaming insane amount of data about the car and its surroundings
in (almost) real time, from the temperature of the limited slip differential to
the tire wear.
Each races however, contain many analog properties which are hard to predict of which
the weather is one of the most important. Teams comes out with different
strategies to counter them. They have in 2017, a choice of 7 different tires types and
5 different fuel types.
In parallel the regulations constantly evolve to make sure that team with lower budget
can still compete. This year, amongst the many changes, the tires are 25% larger
and the engines more powerful.

The idea behind this project is, are we able, to predict, with a reasonable error rate
the performance of each driver, by adjusting for the weather, the driver, the circuits
properties and the qualifications times.


# The iteration modeling process

The model has been built following the CRISP-DM process (https://en.wikipedia.org/wiki/Cross_Industry_Standard_Process_for_Data_Mining).

Once the data - understanding/cleaning phase is done the iteration process
constantly highlight how it essential to constantly review the business understanding
after each iteration. By doing this I discovered properties that did not sound important
at first (like the direction (clockwise or anti-clockwise) or the type of the circuit
Street/Road/Race circuit).


# Getting the data

The first and main source of the data comes from the Ergast motor racing results API (http://ergast.com) maintained by Chris Newell which summarize data from 1950 onwards.

External, data sources were added along the way using unstructured sources like @f1debrief
who posted lap-times and tires compounds for 100 long runs on Twitter.


# Understanding the signals

One of the challenge has been understanding which are the key components which are involved
with the process of a driver making good time. Overall in this model they can be
grouped, by:
- the team adjustments
- the drivers' performance
- the competition effect
- the circuits' properties
- the regulation Overall effect / year
- the pit stops strategies

# Modeling against the residuals / GradientBoosting

The natural unpredictability of a driver not finishing a race is on its own a
very hard problem to solve.
Here we account for the probability of the combination driver/team of not finishing
the race using a second model applied on the residuals.
