OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[4], q[3];
rz(0.9646973041833197) q[1];
rz(4.756525998327407) q[2];
rz(4.221775438312858) q[0];
cx q[1], q[3];
rz(1.4306222832278581) q[0];
rz(3.5881054650582476) q[4];
rz(0.35761051187776305) q[2];
rz(5.649866230449149) q[3];
rz(2.3777909989608244) q[2];
rz(4.544134011986054) q[4];
rz(2.8426920911544764) q[0];
rz(5.418976832009295) q[1];
rz(4.401722775506363) q[3];
rz(3.3598095369437098) q[4];
cx q[0], q[1];
rz(4.13837112221355) q[2];
cx q[3], q[0];
cx q[1], q[4];
rz(6.240520068536379) q[2];
cx q[1], q[4];
rz(1.9181497023069471) q[0];
cx q[2], q[3];
rz(5.4312198944901295) q[1];
rz(4.555158986418246) q[3];