OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
x q[0];
x q[1];
x q[2];
x q[3];
x q[6];
x q[0];
h q[0];
rxx(0.4692358374595642) q[0], q[9];
rxx(0.8995885848999023) q[1], q[9];
rxx(0.3943318724632263) q[2], q[9];
rxx(0.08269613981246948) q[3], q[9];
rxx(0.7410070896148682) q[4], q[9];
rxx(0.31226658821105957) q[5], q[9];
rxx(0.505364716053009) q[6], q[9];
rxx(0.9298731684684753) q[7], q[9];
rxx(0.691366970539093) q[8], q[9];
rzz(0.0012799501419067383) q[0], q[9];
rzz(0.07924222946166992) q[1], q[9];
rzz(0.7572689652442932) q[2], q[9];
rzz(0.15439116954803467) q[3], q[9];
rzz(0.4241400957107544) q[4], q[9];
rzz(0.11954689025878906) q[5], q[9];
rzz(0.7961922883987427) q[6], q[9];
rzz(0.735521137714386) q[7], q[9];
rzz(0.1418662667274475) q[8], q[9];
rzx(0.7799512147903442) q[0], q[9];
rzx(0.13929849863052368) q[1], q[9];
rzx(0.623687207698822) q[2], q[9];
rzx(0.09409844875335693) q[3], q[9];
rzx(0.011251688003540039) q[4], q[9];
rzx(0.9529641270637512) q[5], q[9];
rzx(0.5673917531967163) q[6], q[9];
rzx(0.10637772083282471) q[7], q[9];
rzx(0.49010568857192993) q[8], q[9];
h q[0];
