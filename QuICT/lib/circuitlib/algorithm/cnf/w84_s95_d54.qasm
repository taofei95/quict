OPENQASM 2.0;
include "qelib1.inc";
qreg q[84];
creg c[84];
ccx q[81], q[80], q[82];
x q[25];
x q[70];
x q[81];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
x q[25];
x q[70];
ccx q[81], q[80], q[82];
x q[20];
x q[80];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
x q[20];
ccx q[81], q[80], q[82];
x q[25];
x q[70];
x q[81];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
x q[25];
x q[70];
ccx q[81], q[80], q[82];
x q[20];
x q[80];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
x q[20];
x q[1];
x q[14];
x q[43];
x q[83];
ccx q[43], q[82], q[83];
ccx q[1], q[14], q[82];
ccx q[43], q[82], q[83];
ccx q[1], q[14], q[82];
x q[1];
x q[14];
x q[43];
ccx q[82], q[83], q[80];
ccx q[81], q[80], q[82];
x q[25];
x q[70];
x q[81];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
x q[25];
x q[70];
ccx q[81], q[80], q[82];
x q[20];
x q[80];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
x q[20];
ccx q[81], q[80], q[82];
x q[25];
x q[70];
x q[81];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
ccx q[70], q[83], q[81];
ccx q[25], q[48], q[83];
x q[25];
x q[70];
ccx q[81], q[80], q[82];
x q[20];
x q[80];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
ccx q[66], q[83], q[80];
ccx q[5], q[20], q[83];
x q[20];
x q[1];
x q[14];
x q[43];
x q[83];
ccx q[43], q[82], q[83];
ccx q[1], q[14], q[82];
ccx q[43], q[82], q[83];
ccx q[1], q[14], q[82];
x q[1];
x q[14];
x q[43];
