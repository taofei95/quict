OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rz(2.625186104226855) q[1];
cx q[0], q[3];
cx q[5], q[2];
rz(0.28784159052739194) q[4];
rz(4.038000333145976) q[2];
cx q[1], q[5];
rz(1.0451906097079144) q[4];
rz(6.1681047292839075) q[0];
rz(3.0408965385532247) q[3];
rz(2.5519768864395953) q[0];
rz(0.45695557713043383) q[1];
rz(4.40680311088179) q[5];
rz(2.8650926158925065) q[4];
rz(0.2525135354944094) q[3];
rz(3.7188989208935883) q[2];
rz(5.597016864998542) q[3];
rz(4.672102643855723) q[0];
rz(6.104958482872815) q[5];
rz(3.6302625886554485) q[4];
rz(3.215559731857409) q[1];
rz(2.9897701967512047) q[2];
cx q[1], q[2];
rz(1.2048219437082568) q[3];
rz(5.488414126765703) q[5];
rz(5.397860553397209) q[4];
rz(3.9729455360559025) q[0];
cx q[4], q[2];
cx q[1], q[3];
rz(1.6552035180437636) q[5];
rz(0.4039043888582251) q[0];
rz(5.153024142707752) q[2];
rz(2.6398052445329623) q[0];
rz(2.231457085552922) q[1];
rz(6.179030890772988) q[5];
cx q[4], q[3];
cx q[5], q[3];
cx q[1], q[0];
rz(4.757156482550378) q[2];
rz(1.8690994844228852) q[4];
rz(3.4796663153492133) q[0];
rz(0.26523376569911483) q[5];
cx q[3], q[4];
cx q[2], q[1];
rz(2.483431116016763) q[2];
rz(5.791477723822542) q[0];
rz(4.605689939578785) q[5];
rz(4.19568063657843) q[1];
rz(5.785968485118664) q[3];
rz(3.973075909484274) q[4];
rz(4.731618909623222) q[0];
rz(3.503411530205074) q[1];
rz(0.6024625643383483) q[5];
rz(1.7151111102667154) q[2];
rz(5.01308600081311) q[4];
rz(3.749902148615948) q[3];
rz(5.880839986826394) q[0];
cx q[2], q[4];
rz(2.1231055015415192) q[1];
rz(0.3471590232428796) q[5];
rz(0.3332289852069781) q[3];
cx q[3], q[1];
rz(2.101856424319894) q[0];
rz(0.3045022933504931) q[2];
rz(2.50829075437529) q[5];
rz(4.586291744888078) q[4];
cx q[4], q[0];
rz(5.737884361249994) q[5];
rz(3.4261685073915262) q[1];
rz(4.274928381630335) q[2];
rz(4.7779989059552) q[3];
cx q[4], q[5];
rz(0.5877301118734369) q[1];
cx q[2], q[0];
rz(4.627612667414695) q[3];
rz(1.1392529293797153) q[1];
rz(1.5648204667198848) q[4];
rz(0.7830323061658786) q[3];
cx q[0], q[2];
rz(1.0686287583455572) q[5];
rz(2.2871337334671016) q[0];
cx q[5], q[1];
rz(1.6335848146032466) q[3];
rz(3.640323936642356) q[2];
rz(4.981234789616358) q[4];
rz(5.023316983194436) q[2];
rz(4.500039494179491) q[1];
rz(4.614273713063581) q[4];
rz(3.5733798595192283) q[3];
rz(2.853630422177542) q[5];
rz(4.261933971665326) q[0];
rz(3.928900042086028) q[2];
rz(1.3599564154319155) q[1];
cx q[3], q[4];
rz(2.653208144588123) q[5];
rz(4.724184734953693) q[0];
rz(3.890118112660854) q[4];
rz(3.434977554052301) q[1];
rz(1.1751250998703264) q[0];
rz(5.8710793057243595) q[5];
cx q[3], q[2];
rz(0.8579016556056648) q[3];
rz(2.12403699891585) q[2];
rz(1.237503318021077) q[0];
cx q[4], q[1];
rz(3.254878472865228) q[5];
cx q[1], q[3];
rz(3.8095859216201595) q[2];
rz(4.6948581731487184) q[0];
rz(4.117199257690816) q[4];
rz(3.589364901334383) q[5];
rz(1.5234785265090403) q[3];
rz(2.442097154825255) q[1];
cx q[0], q[5];
rz(0.623544818808925) q[2];
rz(4.544037070492864) q[4];
rz(5.283139738614753) q[0];
cx q[5], q[1];
rz(2.728001946179896) q[4];
rz(5.726251892980207) q[2];
rz(3.1543298439055714) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
