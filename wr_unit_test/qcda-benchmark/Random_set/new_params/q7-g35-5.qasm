OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
p(4.989180032001489) q[2];
tdg q[0];
u1(5.616011125590452) q[0];
h q[6];
u3(1.2680013906802203, 3.916842348830731, 3.944361430349473) q[6];
u2(2.3408410727268367, 5.127813357574432) q[4];
u3(1.4304029350285798, 2.6318684191751647, 2.6313242928792127) q[2];
u3(5.916101469741815, 4.574356614454137, 5.207868584182483) q[0];
rz(3.1414716388972868) q[1];
t q[6];
sdg q[6];
id q[3];
ryy(3.0598575844174585) q[1], q[3];
x q[3];
ry(1.427353246096205) q[2];
sdg q[0];
id q[4];
u2(1.5439244530362974, 3.8173991116108774) q[2];
h q[1];
s q[5];
s q[2];
tdg q[5];
p(0.8408220389299931) q[3];
s q[4];
u2(0.6662652138331709, 0.3021694889089672) q[5];
h q[5];
u3(5.729216445721896, 4.011732349772561, 3.155976361222557) q[0];
sdg q[5];
u2(5.0216480378261705, 1.956615537057707) q[2];
crz(2.4306737260060194) q[4], q[6];
cx q[5], q[6];
rz(2.248297268406229) q[3];
u3(1.1879469816673405, 0.21828932521297742, 3.3518265191408676) q[3];
rxx(2.0914155649555797) q[4], q[6];
t q[4];