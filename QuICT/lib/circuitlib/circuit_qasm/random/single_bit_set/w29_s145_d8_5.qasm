OPENQASM 2.0;
include "qelib1.inc";
qreg q[29];
creg c[29];
rz(1.5707963267948966) q[11];
rx(1.5707963267948966) q[15];
s q[5];
ry(1.5707963267948966) q[24];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[3];
t q[22];
t q[14];
h q[12];
t q[11];
x q[15];
u3(0, 0, 1.5707963267948966) q[25];
s q[17];
u1(1.5707963267948966) q[8];
t q[8];
u3(0, 0, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[19];
h q[1];
u1(1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[23];
t q[1];
tdg q[5];
sdg q[7];
z q[23];
u1(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[28];
z q[4];
h q[16];
rz(1.5707963267948966) q[2];
y q[20];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[27];
z q[27];
y q[18];
y q[15];
z q[4];
h q[12];
y q[4];
z q[2];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[17];
sdg q[21];
u3(0, 0, 1.5707963267948966) q[12];
tdg q[0];
x q[2];
rz(1.5707963267948966) q[11];
h q[13];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[2];
h q[13];
y q[27];
h q[4];
h q[7];
y q[26];
y q[19];
x q[17];
tdg q[5];
s q[5];
rz(1.5707963267948966) q[11];
s q[24];
h q[7];
s q[17];
tdg q[16];
t q[24];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[23];
x q[3];
sdg q[18];
x q[18];
u3(0, 0, 1.5707963267948966) q[22];
t q[0];
sdg q[13];
tdg q[14];
y q[23];
x q[3];
t q[1];
z q[10];
t q[5];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[28];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[19];
sdg q[8];
rx(1.5707963267948966) q[3];
x q[14];
tdg q[16];
tdg q[10];
u1(1.5707963267948966) q[12];
sdg q[25];
t q[20];
z q[16];
tdg q[6];
s q[10];
h q[12];
z q[27];
s q[20];
y q[1];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[27];
rx(1.5707963267948966) q[15];
x q[18];
z q[11];
h q[13];
s q[15];
y q[26];
y q[22];
s q[20];
u1(1.5707963267948966) q[6];
x q[9];
ry(1.5707963267948966) q[24];
t q[20];
y q[8];
s q[23];
h q[22];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[6];
t q[28];
z q[12];
ry(1.5707963267948966) q[15];
x q[5];
s q[18];
x q[25];
sdg q[0];
ry(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[21];
rz(1.5707963267948966) q[24];
sdg q[6];
ry(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[19];
tdg q[11];
rx(1.5707963267948966) q[4];
