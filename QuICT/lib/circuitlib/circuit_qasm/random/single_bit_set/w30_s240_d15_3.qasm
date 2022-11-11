OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
z q[22];
rz(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[2];
u3(0, 0, 1.5707963267948966) q[12];
tdg q[21];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
t q[2];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[23];
t q[26];
s q[29];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[29];
z q[16];
tdg q[22];
u3(0, 0, 1.5707963267948966) q[9];
t q[4];
y q[21];
rz(1.5707963267948966) q[26];
u2(1.5707963267948966, 1.5707963267948966) q[13];
x q[27];
u1(1.5707963267948966) q[24];
tdg q[11];
z q[12];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
h q[11];
y q[14];
z q[2];
sdg q[5];
u1(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[5];
tdg q[7];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[20];
h q[12];
u3(0, 0, 1.5707963267948966) q[22];
tdg q[29];
sdg q[4];
sdg q[13];
z q[21];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[27];
s q[19];
u2(1.5707963267948966, 1.5707963267948966) q[15];
sdg q[5];
z q[22];
s q[5];
t q[9];
s q[14];
rz(1.5707963267948966) q[1];
z q[22];
u1(1.5707963267948966) q[20];
t q[14];
sdg q[17];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[9];
x q[2];
h q[16];
rx(1.5707963267948966) q[27];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[18];
y q[13];
z q[5];
sdg q[22];
ry(1.5707963267948966) q[19];
t q[22];
rz(1.5707963267948966) q[6];
h q[3];
u3(0, 0, 1.5707963267948966) q[20];
sdg q[20];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[3];
z q[26];
s q[25];
s q[23];
sdg q[18];
s q[7];
t q[21];
z q[28];
sdg q[12];
u1(1.5707963267948966) q[16];
rz(1.5707963267948966) q[21];
rx(1.5707963267948966) q[23];
z q[8];
t q[25];
u3(0, 0, 1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[22];
rx(1.5707963267948966) q[13];
h q[23];
sdg q[21];
ry(1.5707963267948966) q[27];
rx(1.5707963267948966) q[17];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[5];
sdg q[20];
sdg q[10];
z q[2];
sdg q[24];
u2(1.5707963267948966, 1.5707963267948966) q[20];
ry(1.5707963267948966) q[2];
z q[4];
u1(1.5707963267948966) q[11];
sdg q[17];
u3(0, 0, 1.5707963267948966) q[16];
tdg q[23];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[12];
z q[17];
h q[2];
u1(1.5707963267948966) q[9];
h q[22];
h q[6];
tdg q[10];
rz(1.5707963267948966) q[23];
s q[18];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[13];
y q[12];
h q[21];
sdg q[0];
t q[17];
rx(1.5707963267948966) q[17];
tdg q[0];
tdg q[18];
tdg q[28];
rx(1.5707963267948966) q[3];
t q[7];
y q[11];
ry(1.5707963267948966) q[9];
u1(1.5707963267948966) q[21];
tdg q[25];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[21];
y q[0];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
sdg q[15];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[10];
u1(1.5707963267948966) q[29];
tdg q[10];
rz(1.5707963267948966) q[25];
t q[28];
h q[6];
s q[22];
z q[24];
x q[4];
u3(0, 0, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[15];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[15];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[12];
ry(1.5707963267948966) q[6];
sdg q[9];
tdg q[23];
u1(1.5707963267948966) q[0];
tdg q[4];
h q[16];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[3];
y q[0];
t q[11];
u3(0, 0, 1.5707963267948966) q[14];
u1(1.5707963267948966) q[26];
tdg q[28];
sdg q[0];
x q[2];
x q[18];
rz(1.5707963267948966) q[23];
y q[18];
ry(1.5707963267948966) q[7];
h q[12];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[22];
sdg q[4];
z q[14];
sdg q[0];
h q[0];
sdg q[6];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[6];
z q[28];
h q[4];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[5];
x q[22];
sdg q[4];
h q[25];
tdg q[25];
ry(1.5707963267948966) q[8];
s q[14];
u2(1.5707963267948966, 1.5707963267948966) q[2];
y q[10];
y q[7];
t q[7];
tdg q[28];
h q[8];
sdg q[29];
x q[28];
ry(1.5707963267948966) q[22];
rx(1.5707963267948966) q[29];
s q[11];
y q[2];
u2(1.5707963267948966, 1.5707963267948966) q[21];
t q[28];
tdg q[17];
t q[22];
t q[5];
t q[11];
x q[3];
ry(1.5707963267948966) q[19];
tdg q[24];
h q[6];
x q[15];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[28];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[11];
t q[17];
rx(1.5707963267948966) q[13];
t q[13];
