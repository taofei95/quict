OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
cx q[2], q[4];
rz(2.6980580916586505) q[7];
rz(3.3708068197672545) q[8];
rz(4.169935604516137) q[5];
rz(3.5764785177101004) q[11];
cx q[1], q[10];
rz(0.5730660645404143) q[3];
rz(0.9224204776161486) q[0];
rz(5.871406896436073) q[9];
rz(2.7343594001666696) q[6];
cx q[11], q[10];
rz(5.994551294789879) q[7];
cx q[3], q[8];
rz(5.013062725159223) q[6];
rz(4.693890353480054) q[5];
rz(3.115311445945611) q[1];
rz(0.6752588850494994) q[9];
rz(0.09825199419358056) q[0];
cx q[2], q[4];
rz(0.9471969414535266) q[8];
rz(5.396861210731303) q[3];
rz(5.365144996499609) q[9];
rz(1.2298386975256912) q[0];
rz(3.482348682613885) q[10];
rz(4.3263692421474875) q[2];
cx q[11], q[5];
rz(4.63614251701736) q[7];
rz(3.383449604613934) q[6];
rz(5.447206773285939) q[4];
rz(2.353013617825805) q[1];
cx q[11], q[3];
rz(2.948796107678936) q[4];
rz(0.32992153677125535) q[2];
rz(2.066101755615535) q[0];
rz(0.7438944843554842) q[8];
cx q[1], q[7];
rz(2.23077800483179) q[10];
rz(0.5879534251919374) q[9];
cx q[6], q[5];
rz(2.2796300257865165) q[1];
rz(3.8830709518603452) q[0];
rz(0.5204380160683175) q[7];
rz(4.590811321982753) q[11];
rz(2.4120710768331484) q[10];
cx q[4], q[2];
rz(6.002366495693867) q[3];
cx q[5], q[9];
rz(4.199958223575766) q[8];
rz(3.685118556148802) q[6];
rz(3.3682373979607685) q[7];
rz(2.1256489805858) q[6];
cx q[2], q[0];
cx q[5], q[4];
rz(0.11605099742533974) q[9];
rz(4.554643192385871) q[3];
rz(4.964651149145045) q[8];
rz(3.1796030143173977) q[11];
cx q[10], q[1];
rz(2.6826363657463306) q[3];
cx q[8], q[10];
rz(4.9943243561821955) q[0];
rz(4.681387997165431) q[5];
rz(1.0621538297388358) q[7];
rz(5.702626472509196) q[11];
rz(2.804386773476304) q[2];
cx q[1], q[9];
rz(3.6545488268758333) q[4];
rz(1.487873176400429) q[6];
rz(4.235824327112136) q[1];
rz(3.914547987458115) q[10];
cx q[0], q[8];
rz(2.8855695232013994) q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];