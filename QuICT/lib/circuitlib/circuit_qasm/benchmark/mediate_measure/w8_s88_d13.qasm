OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rz(3.148104425002611) q[6];
rz(1.9248612370864118) q[2];
rz(2.40482129109751) q[4];
rz(5.69761349890385) q[1];
rz(2.1728119322450428) q[5];
rz(3.11795906114404) q[7];
rz(1.1969235639215863) q[3];
rz(0.5566546489708077) q[0];
rz(4.79981496865731) q[6];
rz(4.6801638024628) q[3];
cx q[4], q[1];
rz(1.220305369972777) q[0];
cx q[2], q[5];
rz(1.4310074286232188) q[7];
rz(1.7404763114458817) q[3];
rz(0.5321295387628926) q[0];
rz(5.728364333740463) q[1];
rz(0.08481570356274619) q[7];
rz(1.1558763553404552) q[4];
cx q[6], q[5];
rz(2.585295096089677) q[2];
rz(4.3688935039008685) q[1];
rz(0.9728797079034608) q[2];
rz(4.7039477194950985) q[0];
rz(6.235450782780742) q[6];
rz(4.333890432847229) q[5];
rz(3.272361383282066) q[3];
rz(5.319001506663945) q[4];
rz(2.0023061622686944) q[7];
rz(2.3297918784887877) q[1];
rz(1.2673481838113503) q[0];
rz(2.71415159978207) q[5];
cx q[4], q[7];
rz(1.7278728441635596) q[3];
rz(3.7271737750820755) q[2];
rz(5.856848131254754) q[6];
rz(2.0041870906399817) q[5];
rz(1.4245657413345474) q[1];
rz(0.5091260074078295) q[3];
rz(1.755774743276872) q[2];
rz(4.772063764874868) q[6];
rz(0.18680719003899232) q[4];
rz(5.037554484675535) q[0];
rz(0.6343124397928895) q[7];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
rz(0.0828893353343686) q[4];
cx q[0], q[6];
rz(1.3154375724914482) q[2];
rz(0.15650655060612317) q[1];
rz(3.445475886772216) q[5];
rz(2.3935401617244243) q[7];
rz(1.5317342183375788) q[3];
cx q[7], q[3];
rz(4.967621197662396) q[0];
rz(2.35706711728388) q[2];
rz(5.723320390005209) q[6];
rz(4.433738044128636) q[4];
cx q[5], q[1];
rz(0.6139304464765805) q[2];
rz(2.809242405661635) q[6];
cx q[5], q[1];
rz(1.2530885961893825) q[0];
rz(5.860753565325637) q[7];
cx q[3], q[4];
rz(2.318104756797259) q[2];
cx q[3], q[1];
rz(5.820810870873333) q[5];
rz(3.55340790133348) q[7];
rz(1.5958030087425068) q[4];
rz(2.582106369428475) q[0];
rz(1.5661237098307406) q[6];
rz(4.526674651256012) q[3];
rz(3.171643570635261) q[1];
cx q[7], q[4];
rz(3.0584590669777763) q[5];
rz(3.3461116459098474) q[2];
cx q[6], q[0];
rz(1.3514578068193897) q[7];
rz(3.3533603076999996) q[0];
rz(3.7239005971779995) q[3];
rz(3.9032363555846814) q[5];
