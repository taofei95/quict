OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
sdg q[1];
s q[4];
u3(3.969499774545558, 0.3776020781107706, 3.7237263385561086) q[4];
cu3(0.7977663447804385, 2.3300394049336486, 4.816481229247975) q[3], q[2];
ry(3.743200661004538) q[0];
p(5.739975675055987) q[2];
h q[3];
h q[1];
u2(6.030404588949598, 3.753531541986428) q[4];
h q[3];
tdg q[4];
u2(2.6152497098089658, 1.3674253955155973) q[3];
sdg q[1];
id q[2];
tdg q[3];
t q[4];
rz(4.5365867343678286) q[2];
ry(5.772985614716058) q[4];
s q[3];
rzz(5.1422948840777964) q[1], q[3];
cu1(0.5463506912930509) q[3], q[4];
p(1.2706881755470663) q[1];
sdg q[4];
ryy(0.3206311528876992) q[1], q[3];
h q[2];
h q[3];
rx(1.5732320597722425) q[4];
tdg q[0];
p(1.586630787651404) q[4];
rz(5.214644608122939) q[4];
tdg q[2];
rzz(3.176176101898319) q[3], q[2];
x q[4];
rzz(1.2779421534402937) q[1], q[0];
h q[3];
t q[0];
ry(4.190933493618892) q[0];
u3(3.846030533996114, 3.2917879762257423, 4.963314417112413) q[2];
rx(0.742797176868165) q[4];
rx(4.5324501274195095) q[2];
s q[3];
rxx(2.921669717998403) q[0], q[2];
rzz(0.03614389471474186) q[3], q[2];
ch q[2], q[3];
t q[3];
sdg q[3];
rzz(2.6268266940241287) q[2], q[1];
x q[4];
t q[2];
t q[4];
cy q[2], q[1];
cx q[2], q[4];
tdg q[1];
rz(5.94736801523528) q[1];
u1(0.7145952946929727) q[0];
t q[2];
u3(0.03107994580294263, 4.196244268786403, 6.028232904525375) q[4];
ch q[1], q[4];
sdg q[0];
cu1(6.067187919413167) q[1], q[3];
u1(4.057290942823454) q[4];
rz(4.732548648962336) q[3];
t q[2];
rz(5.945173828758362) q[3];
rxx(1.7147056668633933) q[3], q[4];