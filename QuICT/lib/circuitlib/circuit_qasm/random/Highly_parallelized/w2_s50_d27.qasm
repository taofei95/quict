OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(0.6138601176118574) q[1];
rz(2.810238280389838) q[0];
rz(5.939190842416461) q[0];
rz(3.3447812732747373) q[1];
rz(5.822294072367161) q[0];
rz(0.8744838318749794) q[1];
rz(0.6349713084926102) q[1];
rz(4.6400470244039305) q[0];
rz(5.860494725812775) q[0];
rz(2.988520699554052) q[1];
cx q[0], q[1];
rz(2.5570510680582266) q[0];
rz(2.1041466078810225) q[1];
rz(5.293997888856796) q[0];
rz(6.263670135655292) q[1];
rz(4.101027320815007) q[0];
rz(1.9346646605939863) q[1];
cx q[1], q[0];
rz(0.7779151441466986) q[1];
rz(0.922598812359164) q[0];
rz(5.07793457395843) q[0];
rz(4.9623213868050176) q[1];
rz(0.2893637037704487) q[0];
rz(1.18603149866284) q[1];
rz(2.5010463774562295) q[0];
rz(3.017354099737966) q[1];
rz(2.20788160405504) q[0];
rz(1.155285054085305) q[1];
rz(2.6936636788511055) q[0];
rz(0.9398525106470917) q[1];
rz(6.083370004065541) q[1];
rz(0.021200783691095265) q[0];
rz(1.2120139639745298) q[1];
rz(6.192643676207528) q[0];
rz(4.3423104788391935) q[0];
rz(0.08691210027765986) q[1];
rz(1.6357689128719026) q[0];
rz(1.4471193085595655) q[1];
rz(6.239995524873347) q[1];
rz(3.133158478554546) q[0];
cx q[0], q[1];
rz(3.709296429284921) q[1];
rz(0.4546676368819429) q[0];
cx q[1], q[0];
rz(3.901336950191876) q[1];
rz(4.06331972810464) q[0];
rz(3.4455946902387127) q[0];
rz(2.1389778951595595) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];