OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
id q[2];
sdg q[10];
id q[5];
id q[2];
sdg q[6];
u1(1.5707963267948966) q[18];
id q[2];
s q[3];
s q[1];
t q[14];
tdg q[16];
id q[3];
u1(1.5707963267948966) q[18];
id q[17];
id q[1];
z q[7];
sdg q[5];
t q[6];
z q[10];
t q[18];
sdg q[6];
sdg q[2];
rz(1.5707963267948966) q[16];
tdg q[3];
t q[16];
sdg q[18];
sdg q[1];
sdg q[16];
rz(1.5707963267948966) q[9];
id q[7];
tdg q[14];
sdg q[10];
t q[9];
rz(1.5707963267948966) q[16];
id q[17];
z q[10];
id q[7];
sdg q[9];
tdg q[9];
t q[5];
rz(1.5707963267948966) q[1];
s q[7];
id q[11];
z q[7];
sdg q[6];
id q[10];
tdg q[14];
z q[16];
z q[9];
s q[8];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[7];
id q[12];
s q[12];
rz(1.5707963267948966) q[17];
t q[2];
id q[8];
z q[18];
rz(1.5707963267948966) q[7];
tdg q[12];
s q[17];
tdg q[13];
t q[3];
id q[9];
u1(1.5707963267948966) q[16];
s q[1];
rz(1.5707963267948966) q[1];
u1(1.5707963267948966) q[13];
tdg q[5];
sdg q[18];
id q[4];
id q[12];
u1(1.5707963267948966) q[17];
s q[11];
s q[11];
s q[4];
z q[17];
z q[10];
tdg q[0];
s q[0];
s q[10];
rz(1.5707963267948966) q[18];
t q[6];
z q[11];
z q[7];
s q[14];
id q[3];
rz(1.5707963267948966) q[2];
z q[11];
id q[16];
z q[5];
sdg q[6];
rz(1.5707963267948966) q[7];
t q[15];
tdg q[8];
t q[8];
sdg q[3];
u1(1.5707963267948966) q[18];
sdg q[9];
t q[13];
t q[15];
sdg q[14];
sdg q[2];
s q[3];
s q[10];
z q[5];
u1(1.5707963267948966) q[5];
sdg q[12];
u1(1.5707963267948966) q[8];
t q[3];
t q[15];
t q[3];
id q[12];
