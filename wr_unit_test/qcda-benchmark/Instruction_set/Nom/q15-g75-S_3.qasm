OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
h q[7];
cx q[2], q[10];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
h q[10];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[14];
h q[14];
h q[7];
h q[6];
cx q[11], q[14];
cx q[14], q[5];
h q[11];
h q[13];
h q[7];
rz(1.5707963267948966) q[3];
cx q[3], q[12];
rz(1.5707963267948966) q[10];
h q[11];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[11];
h q[10];
rz(1.5707963267948966) q[1];
rz(1.5707963267948966) q[14];
cx q[13], q[12];
rz(1.5707963267948966) q[0];
cx q[8], q[14];
h q[5];
cx q[3], q[9];
h q[12];
h q[1];
cx q[8], q[4];
cx q[3], q[12];
h q[6];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[13];
h q[14];
rz(1.5707963267948966) q[6];
h q[1];
rz(1.5707963267948966) q[5];
cx q[7], q[4];
h q[6];
cx q[12], q[0];
cx q[13], q[7];
h q[12];
cx q[8], q[10];
cx q[11], q[10];
h q[0];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[12];
cx q[1], q[6];
rz(1.5707963267948966) q[11];
h q[14];
cx q[0], q[6];
rz(1.5707963267948966) q[0];
h q[4];
h q[8];
cx q[14], q[12];
h q[5];
rz(1.5707963267948966) q[12];
h q[12];
rz(1.5707963267948966) q[11];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
cx q[6], q[7];
rz(1.5707963267948966) q[12];
cx q[11], q[3];
rz(1.5707963267948966) q[4];
cx q[1], q[13];
h q[9];
rz(1.5707963267948966) q[0];
cx q[0], q[1];
