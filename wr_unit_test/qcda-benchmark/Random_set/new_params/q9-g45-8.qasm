OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
u2(0.23837447956307944, 4.230730280828947) q[2];
rz(0.10416456829256937) q[0];
sdg q[6];
s q[3];
p(0.8452342790271589) q[3];
t q[0];
ry(6.138433726200212) q[6];
cz q[0], q[4];
s q[7];
rx(1.8321387473689568) q[5];
ry(3.3320226142705147) q[6];
s q[0];
u1(4.491648056760234) q[6];
t q[8];
p(3.1169726303564804) q[7];
x q[3];
u2(1.1660662482816915, 2.3077981262960083) q[2];
swap q[5], q[6];
cx q[6], q[2];
id q[6];
u2(3.879322326312787, 2.526846952646605) q[2];
u2(5.162266108482788, 5.507087640750757) q[6];
cx q[0], q[4];
ch q[0], q[3];
s q[3];
swap q[3], q[4];
tdg q[6];
cu3(2.475960175600699, 3.756001713095487, 1.4339009566672558) q[6], q[1];
s q[5];
u2(5.801889257209199, 1.0792532151695715) q[6];
x q[3];
u2(6.206857403294681, 1.9046470749182896) q[1];
x q[5];
u1(1.708293202429787) q[3];
rz(0.5950283661649788) q[7];
cy q[7], q[1];
t q[1];
sdg q[2];
rz(5.710164897747471) q[3];
sdg q[0];
x q[6];
tdg q[8];
s q[7];
u3(5.733499052432883, 1.9128414610574656, 5.202971782264574) q[8];
p(4.427607496636813) q[2];