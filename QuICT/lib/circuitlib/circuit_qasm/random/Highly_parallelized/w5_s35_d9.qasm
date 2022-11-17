OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[4], q[0];
rz(1.32604558957523) q[3];
cx q[1], q[2];
cx q[4], q[3];
cx q[2], q[0];
rz(4.26751165048226) q[1];
cx q[0], q[4];
rz(5.294322704437487) q[3];
rz(0.3456788960257023) q[2];
rz(4.454670986330509) q[1];
rz(0.057793341806508275) q[1];
rz(0.15052207276514643) q[4];
rz(4.201381374245638) q[3];
rz(5.298856210835277) q[2];
rz(3.2279322872527625) q[0];
cx q[4], q[2];
rz(1.7018731157074376) q[1];
cx q[0], q[3];
rz(1.7861926463772828) q[4];
rz(5.191165252969013) q[2];
rz(4.81394119574839) q[0];
rz(4.811047485875218) q[1];
rz(5.838636300121029) q[3];
cx q[3], q[1];
rz(2.301519375515563) q[2];
cx q[4], q[0];
cx q[4], q[2];
rz(2.206670247615326) q[0];
rz(0.6349717552569124) q[1];
rz(4.494115369028308) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];