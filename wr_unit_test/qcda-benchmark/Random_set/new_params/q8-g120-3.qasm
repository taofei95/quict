OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
ch q[0], q[2];
rx(5.320178117714586) q[0];
p(5.4068148876023905) q[2];
u2(2.7277132097076104, 2.00277073821389) q[0];
crz(3.1618198693340216) q[0], q[1];
u3(4.85466442410233, 3.8686018757149006, 4.036696832265937) q[7];
sdg q[6];
cx q[6], q[7];
id q[6];
tdg q[6];
tdg q[5];
x q[0];
ry(0.3175424619412357) q[3];
ry(2.4240429101793546) q[1];
tdg q[6];
u3(4.139684627525095, 5.634340581676641, 6.01155959823445) q[3];
ry(6.118310373139659) q[7];
rzz(3.0776640883833144) q[0], q[3];
p(1.0138504541205258) q[6];
id q[6];
t q[3];
u3(3.412280788604163, 6.1015742403365945, 2.00958352508015) q[0];
u2(5.5887259616690885, 5.671335475463519) q[2];
sdg q[2];
id q[2];
crz(2.3959201802629098) q[0], q[5];
u3(1.3006958901964838, 5.637856339112361, 1.515784604367014) q[5];
cu1(4.371823536977759) q[0], q[3];
s q[3];
u2(5.730793563143337, 3.0470710359946356) q[2];
ch q[0], q[3];
sdg q[6];
ry(5.480064285312056) q[1];
s q[7];
u2(2.7928493340170215, 0.46520363480391097) q[2];
u3(4.300378305432651, 5.0016228351327205, 1.417672257024742) q[0];
u2(2.703804063517188, 1.216799198823083) q[5];
ry(3.753144021003477) q[5];
sdg q[3];
rx(1.0423738501444506) q[2];
ryy(2.617191287116768) q[1], q[4];
id q[7];
id q[7];
sdg q[5];
id q[1];
u2(3.0598418492984334, 5.228411518453026) q[2];
ry(5.902021698203251) q[1];
ch q[7], q[1];
ryy(1.248735025967673) q[0], q[7];
u2(4.863126361948546, 4.611982645258335) q[5];
p(2.7074436531300687) q[4];
tdg q[6];
cu1(4.637792315581671) q[5], q[7];
t q[4];
t q[6];
u1(2.707684321722521) q[2];
rz(4.290984551940306) q[5];
cu3(4.75593616829386, 0.20917760511591174, 4.222221364029221) q[1], q[6];
ryy(4.649248492308997) q[0], q[3];
cu3(3.81020968618562, 2.9171394411628286, 1.8906286418494636) q[6], q[5];
tdg q[4];
crz(0.1061061065793412) q[2], q[7];
rx(1.574275174156308) q[2];
u2(2.359609059091753, 1.7581754319927196) q[4];
rzz(5.437027049219085) q[3], q[1];
rx(2.0118749724299407) q[2];
tdg q[3];
s q[3];
u1(3.887075254980623) q[7];
t q[6];
t q[4];
cy q[2], q[0];
ryy(0.7208927954788518) q[0], q[3];
t q[7];
cx q[4], q[5];
p(3.753729641128921) q[2];
sdg q[4];
rx(5.947282514852936) q[3];
cz q[3], q[4];
u1(3.946323941677517) q[5];
tdg q[7];
p(4.676641304101678) q[0];
u1(4.136675920425087) q[1];
rx(1.8631158847881273) q[6];
p(5.948349718667664) q[0];
cz q[5], q[7];
u3(2.0752234352499617, 0.0010086668503299296, 0.9600336687388031) q[3];
id q[2];
ry(3.421940564442353) q[4];
cu1(5.563238228282755) q[2], q[3];
u3(0.009199145335137002, 2.4747489660691864, 5.078720467326411) q[3];
cu3(0.5548708019719578, 4.267084338923547, 5.72856253236213) q[0], q[7];
x q[4];
u2(4.322748957040662, 0.8204247668659821) q[0];
rz(2.4596647358849615) q[7];
u2(1.1702322702879329, 1.8379892616786755) q[3];
p(1.0976894900636258) q[6];
id q[1];
x q[1];
s q[0];
t q[4];
u2(5.86103335556031, 5.945222252403817) q[1];
u3(2.211502212532897, 5.409934759157535, 4.367627055731345) q[7];
s q[7];
p(1.2269230502947543) q[3];
u1(0.32195268478451916) q[2];
rz(4.503197437861977) q[1];
rx(4.132512793796287) q[2];
u2(3.4752287100100507, 5.398439395592444) q[6];
sdg q[0];
u1(3.432723343490848) q[5];
sdg q[0];
s q[7];
t q[4];
ryy(2.207512336234715) q[4], q[7];
u3(3.156313126140154, 2.1352383564691713, 3.986644361011214) q[3];
ry(5.873658916388489) q[0];
ry(2.880870809687217) q[0];
sdg q[1];
rz(5.143584386327284) q[7];