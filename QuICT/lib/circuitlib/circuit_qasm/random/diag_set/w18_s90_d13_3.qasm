OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
sdg q[10];
id q[1];
sdg q[0];
id q[12];
s q[5];
sdg q[6];
s q[7];
s q[12];
s q[4];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[7];
sdg q[11];
sdg q[5];
rz(1.5707963267948966) q[16];
z q[11];
z q[1];
u1(1.5707963267948966) q[5];
id q[1];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
id q[7];
t q[13];
tdg q[0];
sdg q[17];
t q[12];
sdg q[9];
tdg q[4];
t q[0];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[16];
id q[17];
rz(1.5707963267948966) q[1];
tdg q[2];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[6];
sdg q[15];
t q[12];
u1(1.5707963267948966) q[2];
z q[13];
u1(1.5707963267948966) q[16];
id q[16];
t q[17];
s q[9];
sdg q[9];
z q[14];
sdg q[11];
z q[15];
z q[1];
sdg q[2];
z q[9];
s q[17];
sdg q[11];
u1(1.5707963267948966) q[9];
u1(1.5707963267948966) q[11];
sdg q[7];
sdg q[14];
id q[2];
id q[12];
z q[4];
id q[13];
id q[0];
t q[10];
sdg q[9];
z q[17];
id q[14];
t q[7];
t q[12];
u1(1.5707963267948966) q[7];
t q[11];
t q[9];
rz(1.5707963267948966) q[15];
t q[11];
u1(1.5707963267948966) q[6];
id q[5];
rz(1.5707963267948966) q[9];
sdg q[1];
t q[10];
t q[16];
t q[13];
sdg q[9];
u1(1.5707963267948966) q[9];
z q[0];
s q[17];
sdg q[16];
z q[15];
sdg q[9];
u1(1.5707963267948966) q[13];
u1(1.5707963267948966) q[15];
sdg q[9];
