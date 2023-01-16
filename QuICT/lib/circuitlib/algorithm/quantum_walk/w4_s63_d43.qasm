OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
h q[3];
h q[0];
cu1(1.5707963267948966) q[1], q[0];
h q[1];
cu1(1.5707963267948966) q[2], q[1];
cu1(0.7853981633974483) q[2], q[0];
h q[2];
cu1(3.141592653589793) q[3], q[2];
cu1(1.5707963267948966) q[3], q[1];
cu1(0.7853981633974483) q[3], q[0];
h q[2];
cu1(-1.5707963267948966) q[2], q[1];
h q[1];
cu1(-0.7853981633974483) q[2], q[0];
cu1(-1.5707963267948966) q[1], q[0];
h q[0];
h q[1];
cu1(1.5707963267948966) q[2], q[1];
h q[2];
cu1(-3.141592653589793) q[3], q[2];
cu1(-1.5707963267948966) q[3], q[1];
h q[2];
cu1(-1.5707963267948966) q[2], q[1];
h q[1];
ccx q[3], q[2], q[1];
cx q[3], q[2];
x q[1];
x q[2];
x q[3];
h q[0];
cu1(1.5707963267948966) q[1], q[0];
h q[1];
cu1(1.5707963267948966) q[2], q[1];
cu1(0.7853981633974483) q[2], q[0];
h q[2];
cu1(3.141592653589793) q[3], q[2];
cu1(1.5707963267948966) q[3], q[1];
cu1(0.7853981633974483) q[3], q[0];
h q[2];
cu1(-1.5707963267948966) q[2], q[1];
h q[1];
cu1(-0.7853981633974483) q[2], q[0];
cu1(-1.5707963267948966) q[1], q[0];
h q[0];
h q[1];
cu1(1.5707963267948966) q[2], q[1];
h q[2];
cu1(-3.141592653589793) q[3], q[2];
cu1(-1.5707963267948966) q[3], q[1];
h q[2];
cu1(-1.5707963267948966) q[2], q[1];
h q[1];
x q[1];
x q[2];
x q[3];
x q[2];
x q[3];
ccx q[3], q[2], q[1];
x q[2];
x q[3];
x q[3];
cx q[3], q[2];
x q[3];