OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
ry(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[3];
s q[8];
x q[5];
rz(1.5707963267948966) q[5];
z q[14];
rz(1.5707963267948966) q[14];
t q[8];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[14];
h q[4];
x q[9];
y q[8];
t q[12];
s q[7];
z q[1];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[14];
y q[9];
rx(1.5707963267948966) q[2];
h q[6];
rx(1.5707963267948966) q[13];
h q[6];
t q[4];
u3(0, 0, 1.5707963267948966) q[12];
s q[13];
rz(1.5707963267948966) q[2];
t q[0];
s q[4];
rx(1.5707963267948966) q[2];
h q[12];
rz(1.5707963267948966) q[8];
z q[6];
tdg q[14];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[2];
x q[13];
rx(1.5707963267948966) q[9];
t q[10];
rz(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
y q[10];
t q[7];
h q[8];
rz(1.5707963267948966) q[11];
h q[11];
x q[10];
y q[0];
u3(0, 0, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
t q[13];
rx(1.5707963267948966) q[10];
s q[2];
y q[9];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[13];
y q[12];
ry(1.5707963267948966) q[11];
h q[12];
tdg q[7];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[2];
rz(1.5707963267948966) q[12];
t q[0];
z q[7];
t q[5];
y q[13];
tdg q[8];
tdg q[14];
h q[0];
