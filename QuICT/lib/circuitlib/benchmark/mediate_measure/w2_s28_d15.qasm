OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(4.403087840349789) q[1];
rz(5.288924124569312) q[0];
rz(0.6686076870298892) q[1];
rz(0.22593107822746145) q[0];
cx q[1], q[0];
rz(6.2130714385877415) q[0];
rz(4.553135501398799) q[1];
cx q[0], q[1];
rz(2.9582879057394442) q[1];
rz(5.440841945845053) q[0];
rz(1.9304478766746713) q[1];
rz(4.293531639391367) q[0];
rz(0.5282073992240492) q[1];
rz(2.597681047180602) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
rz(3.9744340637460946) q[1];
rz(1.709549190265333) q[0];
rz(3.5226079270594277) q[1];
rz(0.6690011585146862) q[0];
rz(3.583499103315259) q[1];
rz(4.012051686463664) q[0];
rz(2.86699701580695) q[0];
rz(0.24171908050755653) q[1];
rz(5.330416834911944) q[1];
rz(1.8777603097330524) q[0];
rz(1.2082428314256786) q[0];
rz(0.2201325237829152) q[1];