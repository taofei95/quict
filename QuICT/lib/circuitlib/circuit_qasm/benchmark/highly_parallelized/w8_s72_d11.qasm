OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
cx q[2], q[6];
rz(2.3866604006846046) q[3];
cx q[5], q[7];
cx q[1], q[4];
rz(2.6665246403440994) q[0];
cx q[6], q[5];
cx q[1], q[3];
cx q[7], q[2];
rz(2.377669643599073) q[4];
rz(5.838621118473452) q[0];
rz(1.0790477531899825) q[6];
rz(2.4545512643007794) q[2];
rz(2.171330101189913) q[5];
rz(4.358229467387326) q[4];
rz(1.4340249184892413) q[3];
rz(2.323627564289028) q[1];
rz(2.5001406752075153) q[0];
rz(1.759206132970408) q[7];
rz(2.9452623886106197) q[7];
rz(3.131516195485352) q[3];
rz(2.6697837578393098) q[2];
rz(4.527282334037826) q[1];
rz(2.301945357351948) q[5];
rz(5.115490821132076) q[0];
cx q[4], q[6];
rz(5.697087007116707) q[4];
cx q[7], q[5];
rz(4.298570335343829) q[1];
cx q[0], q[2];
rz(0.526776441501754) q[6];
rz(4.493019005891091) q[3];
cx q[1], q[2];
rz(1.5650331114796074) q[6];
rz(5.819659893970007) q[3];
rz(1.5160439321134787) q[0];
rz(3.962804934275559) q[5];
rz(2.6227209825874325) q[4];
rz(4.674839313663709) q[7];
rz(5.789104744185325) q[0];
rz(2.6333184251510398) q[6];
rz(1.4898021889326354) q[7];
rz(2.5019484213401557) q[3];
cx q[2], q[1];
rz(1.5678632042613565) q[5];
rz(5.518786936169389) q[4];
rz(5.424566514976947) q[2];
rz(1.3701786912326053) q[1];
rz(4.176495103161177) q[4];
rz(1.1981276893151325) q[3];
rz(3.2330735093280887) q[0];
rz(5.943968364594586) q[7];
rz(1.6420175184920442) q[5];
rz(4.564702533968777) q[6];
cx q[4], q[3];
rz(3.4630248472313694) q[1];
rz(1.841517775684319) q[7];
cx q[6], q[5];
rz(0.46535895455652765) q[0];
rz(0.013367640776837188) q[2];
rz(5.4112294609080855) q[7];
rz(5.099894495602131) q[1];
rz(1.6371645103468346) q[5];
rz(4.291728471233606) q[2];
rz(3.892450166532983) q[4];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];