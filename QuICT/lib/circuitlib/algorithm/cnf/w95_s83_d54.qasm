OPENQASM 2.0;
include "qelib1.inc";
qreg q[95];
creg c[95];
ccx q[92], q[91], q[93];
x q[60];
x q[92];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
x q[60];
ccx q[92], q[91], q[93];
x q[19];
x q[91];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
x q[19];
ccx q[92], q[91], q[93];
x q[60];
x q[92];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
x q[60];
ccx q[92], q[91], q[93];
x q[19];
x q[91];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
x q[19];
x q[19];
x q[30];
x q[94];
ccx q[46], q[93], q[94];
ccx q[19], q[30], q[93];
ccx q[46], q[93], q[94];
ccx q[19], q[30], q[93];
x q[19];
x q[30];
ccx q[93], q[94], q[91];
ccx q[92], q[91], q[93];
x q[60];
x q[92];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
x q[60];
ccx q[92], q[91], q[93];
x q[19];
x q[91];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
x q[19];
ccx q[92], q[91], q[93];
x q[60];
x q[92];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
ccx q[60], q[94], q[92];
ccx q[7], q[20], q[94];
x q[60];
ccx q[92], q[91], q[93];
x q[19];
x q[91];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
ccx q[57], q[94], q[91];
ccx q[19], q[23], q[94];
x q[19];
x q[19];
x q[30];
x q[94];
ccx q[46], q[93], q[94];
ccx q[19], q[30], q[93];
ccx q[46], q[93], q[94];
ccx q[19], q[30], q[93];
x q[19];
x q[30];