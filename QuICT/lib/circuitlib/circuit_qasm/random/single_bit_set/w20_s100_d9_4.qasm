OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
tdg q[5];
u3(0, 0, 1.5707963267948966) q[8];
s q[4];
sdg q[6];
x q[17];
tdg q[19];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[18];
rz(1.5707963267948966) q[8];
h q[12];
z q[8];
z q[3];
rx(1.5707963267948966) q[19];
sdg q[9];
ry(1.5707963267948966) q[4];
t q[19];
sdg q[3];
sdg q[8];
rx(1.5707963267948966) q[1];
y q[15];
u1(1.5707963267948966) q[2];
t q[14];
u2(1.5707963267948966, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[18];
y q[14];
tdg q[16];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[14];
u3(0, 0, 1.5707963267948966) q[3];
s q[0];
rz(1.5707963267948966) q[17];
y q[18];
u3(0, 0, 1.5707963267948966) q[5];
h q[1];
u3(0, 0, 1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rz(1.5707963267948966) q[1];
h q[17];
tdg q[5];
h q[3];
z q[7];
t q[14];
rx(1.5707963267948966) q[3];
t q[4];
s q[19];
y q[5];
ry(1.5707963267948966) q[18];
y q[14];
ry(1.5707963267948966) q[18];
s q[18];
tdg q[15];
ry(1.5707963267948966) q[9];
z q[16];
u2(1.5707963267948966, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[12];
u1(1.5707963267948966) q[15];
rx(1.5707963267948966) q[11];
t q[10];
sdg q[9];
s q[11];
rz(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[9];
x q[3];
u3(0, 0, 1.5707963267948966) q[6];
x q[17];
t q[6];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[5];
t q[11];
h q[3];
t q[17];
u1(1.5707963267948966) q[18];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[12];
z q[15];
z q[19];
t q[7];
z q[8];
y q[13];
u1(1.5707963267948966) q[11];
h q[19];
y q[16];
t q[2];
sdg q[19];
sdg q[7];
rx(1.5707963267948966) q[10];
tdg q[7];
ry(1.5707963267948966) q[18];
h q[3];
rz(1.5707963267948966) q[8];
h q[1];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
tdg q[12];
t q[3];
