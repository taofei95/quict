OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(4.958925006725749) q[1];
rz(2.6484074022681052) q[0];
cx q[4], q[3];
rz(4.597543985016748) q[2];
rz(5.961263937011606) q[4];
rz(0.07887295916332188) q[2];
rz(0.38272511628597516) q[0];
rz(2.7470998128865713) q[3];
rz(4.858000055774906) q[1];
rz(5.9396180615593215) q[3];
rz(2.9852574965150094) q[1];
cx q[4], q[2];
rz(2.5583569209558132) q[0];
cx q[3], q[0];
rz(6.142533758717612) q[1];
cx q[2], q[4];
rz(1.9202615763387443) q[2];
rz(4.274354569899019) q[1];
rz(5.940477829677103) q[4];
rz(3.5009176645539477) q[0];
rz(1.3736046351341153) q[3];
rz(2.089742921994557) q[4];
rz(6.009397357788131) q[0];
cx q[3], q[1];
rz(2.0232196811983187) q[2];
rz(2.6801466730004515) q[4];
rz(0.6774396457427664) q[2];
rz(5.9334548993562395) q[3];
cx q[1], q[0];
rz(5.9726077099927375) q[2];
rz(2.462624290742928) q[0];
cx q[4], q[1];
rz(5.6730711208072115) q[3];
rz(1.5173536657270565) q[1];
rz(0.10257876201758688) q[2];
rz(0.8248441984886608) q[0];
cx q[3], q[4];
rz(1.9604783549680342) q[2];
rz(6.265098795105842) q[1];
rz(4.557474861018619) q[0];
rz(5.711859598299332) q[3];
rz(1.0555256297247189) q[4];
rz(3.3750140213921553) q[0];
cx q[1], q[4];
rz(0.9701650991889923) q[2];
rz(3.4911069138891673) q[3];
rz(2.2127985990077663) q[4];
rz(0.44693032863597715) q[0];
rz(0.668536648861575) q[3];
rz(0.22572583794797751) q[1];
rz(0.9824087784105437) q[2];
cx q[3], q[1];
cx q[0], q[4];
rz(4.157115374179733) q[2];
cx q[0], q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
