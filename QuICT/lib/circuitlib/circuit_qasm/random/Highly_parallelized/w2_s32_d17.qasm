OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(0.8200735867642933) q[0];
rz(3.9374317671711347) q[1];
rz(2.3217720575797784) q[0];
rz(0.6891707885868364) q[1];
rz(0.5350570244392926) q[0];
rz(2.042893686006692) q[1];
rz(5.919115064294904) q[1];
rz(1.171604940491249) q[0];
rz(0.07879772101599934) q[0];
rz(0.7694457016185879) q[1];
cx q[1], q[0];
rz(4.2916618559056285) q[0];
rz(5.184976887982899) q[1];
rz(2.3778327048873216) q[0];
rz(2.8973288919994107) q[1];
rz(1.3561448093917876) q[1];
rz(1.7297451307434186) q[0];
rz(4.230512592033309) q[0];
rz(5.432049875518867) q[1];
rz(5.711465405459953) q[0];
rz(2.670487534143534) q[1];
rz(3.2359427789308537) q[0];
rz(1.7476274835205832) q[1];
rz(3.9158433909123436) q[1];
rz(0.7480470932285506) q[0];
rz(5.452648274660299) q[1];
rz(2.8041550761241907) q[0];
rz(5.86085117756646) q[0];
rz(3.873596961030426) q[1];
rz(1.2140573040856262) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];