OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
s q[13];
u1(1.5707963267948966) q[14];
id q[10];
z q[10];
t q[2];
s q[7];
u1(1.5707963267948966) q[10];
s q[11];
sdg q[14];
s q[7];
id q[12];
t q[16];
z q[2];
tdg q[4];
sdg q[11];
rz(1.5707963267948966) q[6];
z q[2];
tdg q[9];
t q[1];
rz(1.5707963267948966) q[13];
id q[6];
sdg q[11];
rz(1.5707963267948966) q[11];
s q[5];
u1(1.5707963267948966) q[11];
rz(1.5707963267948966) q[14];
id q[13];
z q[13];
id q[7];
tdg q[10];
tdg q[11];
tdg q[13];
sdg q[1];
sdg q[13];
z q[8];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[2];
s q[8];
sdg q[11];
id q[8];
t q[6];
z q[13];
z q[13];
u1(1.5707963267948966) q[10];
s q[16];
u1(1.5707963267948966) q[11];
s q[3];
id q[5];
rz(1.5707963267948966) q[13];
id q[9];
u1(1.5707963267948966) q[3];
z q[1];
s q[11];
sdg q[7];
sdg q[1];
u1(1.5707963267948966) q[0];
id q[7];
t q[0];
tdg q[3];
id q[5];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[14];
id q[10];
u1(1.5707963267948966) q[5];
z q[6];
u1(1.5707963267948966) q[15];
rz(1.5707963267948966) q[4];
id q[5];
sdg q[5];
z q[4];
tdg q[9];
z q[15];
s q[5];
rz(1.5707963267948966) q[9];
id q[12];
u1(1.5707963267948966) q[9];
s q[14];
id q[16];
sdg q[4];
tdg q[15];
tdg q[4];
u1(1.5707963267948966) q[7];
id q[7];
rz(1.5707963267948966) q[5];
t q[14];
id q[15];
t q[11];
t q[5];
z q[13];
s q[15];
z q[11];
tdg q[12];
rz(1.5707963267948966) q[14];
id q[8];
t q[11];
t q[11];
rz(1.5707963267948966) q[9];
t q[6];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[9];
