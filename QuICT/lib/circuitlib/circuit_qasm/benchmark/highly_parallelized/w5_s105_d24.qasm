OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cx q[4], q[0];
rz(0.13575007366463931) q[2];
rz(0.4326465977961767) q[3];
rz(1.4406107108081574) q[1];
rz(4.780521374657037) q[1];
rz(1.5832597191040447) q[2];
rz(3.1464541837258966) q[3];
rz(0.7765753772806029) q[4];
rz(1.448690289185896) q[0];
rz(5.136540583525166) q[1];
rz(2.593745641442477) q[3];
cx q[2], q[0];
rz(0.39202648903625303) q[4];
cx q[4], q[0];
rz(1.7766590606911445) q[1];
rz(3.5537754214704993) q[2];
rz(3.5499096586781262) q[3];
rz(2.914481766964573) q[4];
rz(2.203614040976394) q[3];
rz(4.696761016424854) q[0];
rz(4.717142033923075) q[1];
rz(5.079851146519519) q[2];
rz(3.2005816154305298) q[0];
rz(2.3824931681002695) q[1];
rz(3.5112906655849785) q[4];
rz(4.550514991089471) q[2];
rz(0.8331814045051775) q[3];
rz(5.557318370060609) q[1];
rz(0.7193673036315457) q[2];
rz(5.192964621425659) q[4];
rz(3.8417440176620055) q[0];
rz(0.5615498387787553) q[3];
rz(1.4134803243251746) q[1];
rz(2.763016874335075) q[3];
rz(5.4272757273623675) q[2];
rz(0.547143087793835) q[4];
rz(5.90927483950778) q[0];
rz(2.7263085880474054) q[4];
rz(5.051297153021709) q[1];
cx q[0], q[3];
rz(6.044459583763056) q[2];
rz(2.0788568999093835) q[4];
cx q[2], q[3];
rz(0.875272934088624) q[0];
rz(3.433088502108908) q[1];
rz(4.019094544005442) q[0];
rz(3.9480603418585667) q[2];
rz(2.541513371384636) q[3];
rz(4.818302097370268) q[1];
rz(0.7535081931854997) q[4];
cx q[1], q[3];
rz(6.139086274284113) q[0];
rz(1.0370296764344327) q[2];
rz(0.049846049200564915) q[4];
rz(2.0467813547121043) q[3];
rz(4.1529731954512625) q[2];
rz(5.930722953459812) q[0];
rz(0.7186119226008466) q[1];
rz(0.3957169142769809) q[4];
rz(1.1350950392875034) q[0];
rz(5.364513373201007) q[2];
cx q[1], q[3];
rz(0.6206610537837067) q[4];
rz(1.3354749053936081) q[4];
cx q[1], q[2];
rz(4.8855765789341925) q[0];
rz(4.438229004195504) q[3];
cx q[1], q[0];
rz(5.521726468162061) q[3];
cx q[2], q[4];
rz(5.210495531377746) q[1];
rz(5.71721436787766) q[0];
rz(5.199120671282833) q[2];
rz(1.2993146928813206) q[3];
rz(1.3010934321386052) q[4];
rz(3.2171967960996506) q[0];
rz(2.5660897242530885) q[1];
rz(3.75331107522101) q[3];
rz(2.311336621070414) q[2];
rz(2.9399296760568987) q[4];
rz(0.4582518796007386) q[2];
rz(0.4367623063547351) q[0];
cx q[1], q[4];
rz(2.9799247577000623) q[3];
cx q[1], q[0];
rz(4.42068102305942) q[2];
rz(2.260032601731968) q[4];
rz(0.3109729794981402) q[3];
rz(0.6802091405955304) q[0];
rz(4.822222528472961) q[4];
rz(5.279662036065337) q[3];
rz(5.234325713504586) q[1];
rz(4.251874689517728) q[2];
rz(3.589532048465081) q[2];
rz(3.1437222977734365) q[4];
rz(5.567440386922739) q[3];
rz(2.2533033612476654) q[0];
rz(3.3583478405274536) q[1];
rz(0.8361114807913912) q[3];
rz(2.2193519370208645) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
