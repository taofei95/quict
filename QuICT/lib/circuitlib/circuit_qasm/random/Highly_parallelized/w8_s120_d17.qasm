OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rz(5.939275873634416) q[4];
rz(4.142494289104505) q[0];
rz(5.296199174256514) q[1];
rz(3.52323608134913) q[2];
rz(3.461827122601615) q[5];
rz(1.7072313660175482) q[3];
rz(1.4599850836221688) q[7];
rz(2.8989231768692094) q[6];
rz(4.327078646230425) q[0];
rz(5.451453363898244) q[2];
rz(0.039091896512382235) q[3];
rz(2.2951452802708747) q[5];
rz(3.9906367206174407) q[1];
rz(1.11707018034111) q[6];
rz(1.5025298588045821) q[7];
rz(0.4646109126526321) q[4];
rz(0.28434916985014835) q[2];
rz(2.381411582079625) q[6];
rz(3.324845279331082) q[3];
rz(5.177110700888167) q[7];
cx q[4], q[1];
rz(5.772276535629165) q[0];
rz(5.590328323525487) q[5];
rz(2.563875331173626) q[3];
rz(3.6302651323706705) q[4];
cx q[1], q[0];
cx q[6], q[5];
rz(5.576066795027755) q[2];
rz(3.923104986742839) q[7];
rz(4.54497746298755) q[4];
rz(6.05053993125761) q[7];
rz(2.3511933241132117) q[0];
cx q[1], q[5];
rz(5.417628804003874) q[6];
cx q[2], q[3];
rz(0.7491273554885371) q[0];
cx q[5], q[3];
rz(1.6595415520914765) q[1];
rz(5.386023060204073) q[7];
rz(1.245176417787431) q[2];
rz(1.4680142778533294) q[6];
rz(5.2695229960399965) q[4];
rz(1.0058977265196578) q[7];
rz(1.9106933998876656) q[3];
rz(0.7945379755153743) q[0];
rz(1.7772603256190378) q[1];
rz(1.826158681553725) q[2];
rz(5.198859849895823) q[5];
rz(3.0495457050194976) q[4];
rz(1.7428180953714447) q[6];
rz(3.053363463110167) q[5];
rz(3.11041230433138) q[0];
rz(2.849032648948968) q[6];
rz(5.421918193624402) q[3];
rz(3.1294510616093105) q[7];
rz(0.6902520380809105) q[2];
cx q[4], q[1];
rz(2.541837847886949) q[7];
rz(4.742914175902154) q[1];
rz(1.4395340917479806) q[4];
rz(0.8428022778828488) q[5];
cx q[2], q[6];
rz(2.2747812213675065) q[0];
rz(5.220403938601982) q[3];
rz(2.0075195358469027) q[4];
rz(5.6744763538913485) q[7];
rz(4.120942495921247) q[1];
rz(4.145108622544262) q[5];
rz(1.7984052529063923) q[2];
rz(4.2760086512495326) q[6];
rz(3.3993081591263845) q[3];
rz(3.0450998293164613) q[0];
rz(5.047545671156366) q[4];
rz(6.266609147605746) q[5];
rz(2.323572834498249) q[0];
rz(2.395038304914624) q[2];
rz(5.657084854808437) q[3];
rz(1.2099754499370556) q[6];
rz(4.2749256679907885) q[7];
rz(3.137031323710425) q[1];
rz(4.455164809813112) q[5];
rz(2.0194399277185693) q[6];
rz(5.762458850049492) q[4];
rz(3.273530657287519) q[0];
cx q[1], q[7];
rz(3.900187885371712) q[3];
rz(3.777029636679573) q[2];
cx q[6], q[0];
rz(0.39184324706625245) q[3];
rz(1.5928507165356494) q[7];
cx q[1], q[4];
cx q[2], q[5];
rz(0.49925484744346393) q[0];
rz(2.6736306801261707) q[2];
rz(5.2666892654722) q[1];
rz(2.2013947014032156) q[4];
rz(5.918639691335547) q[3];
cx q[7], q[6];
rz(3.3139254742950546) q[5];
rz(4.883301836199158) q[7];
rz(0.8198923473983866) q[2];
rz(3.931660796190339) q[0];
rz(3.4009138290601553) q[5];
rz(2.3259795762735194) q[3];
rz(0.09222415699496538) q[6];
rz(0.8649132309000505) q[4];
rz(3.6833969784664258) q[1];
rz(1.821829538789942) q[5];
rz(1.5066172135409859) q[2];
rz(5.196655117390621) q[7];
rz(3.7878117158721456) q[3];
rz(2.2617486971580654) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];