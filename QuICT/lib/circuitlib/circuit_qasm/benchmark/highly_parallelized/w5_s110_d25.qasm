OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(6.25945552515913) q[2];
rz(2.655287666854212) q[4];
rz(1.8418835248366103) q[3];
cx q[1], q[0];
rz(2.887688987218673) q[4];
rz(3.5547534586155134) q[3];
rz(3.242271587709643) q[0];
rz(0.2236316006183855) q[2];
rz(2.5851932987643735) q[1];
rz(1.1102049904088533) q[2];
rz(2.2752340912544478) q[1];
rz(2.035072930532982) q[3];
rz(2.3136334935433704) q[0];
rz(5.565488048693286) q[4];
rz(1.7431209140958643) q[1];
rz(3.8485899916525943) q[3];
cx q[2], q[0];
rz(3.509538429245933) q[4];
rz(1.1391839850446168) q[1];
cx q[0], q[4];
rz(3.6864461418501935) q[3];
rz(4.532222097332407) q[2];
rz(3.0421303912226425) q[0];
rz(4.531738427586627) q[2];
rz(4.816970325664515) q[3];
cx q[1], q[4];
rz(2.5529233071420188) q[3];
cx q[4], q[1];
rz(5.275683430525586) q[2];
rz(5.929736396340676) q[0];
cx q[1], q[3];
rz(0.18921755599073706) q[0];
rz(0.49386619514909946) q[4];
rz(2.4061011653621023) q[2];
rz(0.41800489634375576) q[1];
rz(4.789679062569806) q[4];
rz(3.5210728386111456) q[3];
rz(3.2123325211691247) q[2];
rz(0.9076272461710205) q[0];
rz(2.7018621760672072) q[0];
rz(0.5170953019435766) q[1];
rz(4.328757270218848) q[4];
rz(6.121932358646268) q[3];
rz(3.787035431132183) q[2];
rz(4.504925620254173) q[3];
rz(0.6539856979855188) q[1];
rz(1.1883310281303248) q[0];
rz(3.167672638238284) q[2];
rz(3.737525445998233) q[4];
rz(1.7459438102163376) q[2];
rz(5.627149889106721) q[1];
rz(0.6957127778997376) q[4];
cx q[0], q[3];
cx q[0], q[2];
rz(4.796269050769489) q[1];
rz(0.07300845324698435) q[4];
rz(1.0429464159659088) q[3];
rz(3.522893329611228) q[1];
cx q[0], q[4];
rz(3.290444411501379) q[3];
rz(5.7546794199966556) q[2];
rz(2.1268000996245178) q[1];
rz(5.06556190425685) q[2];
rz(0.11842867990248691) q[3];
rz(5.6259100361185865) q[0];
rz(4.057130969163647) q[4];
rz(3.6648664480558093) q[3];
rz(4.107484538087553) q[1];
rz(5.093904554769896) q[0];
rz(1.6609069233773788) q[2];
rz(5.669322989250744) q[4];
rz(4.5776742198730735) q[3];
rz(3.5555376837663046) q[2];
rz(1.1172994928784583) q[1];
cx q[0], q[4];
rz(2.522993725275656) q[4];
rz(5.472836779441408) q[0];
cx q[2], q[3];
rz(5.976974287802145) q[1];
rz(5.124815555620553) q[0];
rz(3.708117778296396) q[1];
rz(0.5978964313540756) q[3];
rz(5.6646698975963545) q[2];
rz(3.916383420315842) q[4];
rz(5.520978397098956) q[3];
rz(2.8699830795383443) q[1];
cx q[2], q[0];
rz(5.09776722590842) q[4];
rz(4.182780063650661) q[1];
rz(5.948126207855558) q[0];
rz(4.666762416246037) q[4];
rz(4.197853087439525) q[3];
rz(0.9805655793198044) q[2];
rz(0.6149844248871242) q[3];
rz(5.028125830339611) q[2];
rz(5.079537938076156) q[0];
rz(0.3977339323218257) q[4];
rz(3.9429402157984104) q[1];
cx q[2], q[0];
rz(0.9216940659042014) q[3];
rz(1.3385009688850642) q[1];
rz(2.6034649788365334) q[4];
rz(0.30296949643179205) q[4];
cx q[2], q[1];
cx q[3], q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];