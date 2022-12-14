OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
rz(4.488739426510892) q[12];
rz(2.8308802070765675) q[2];
rz(2.521740342313334) q[19];
rz(1.337670834401268) q[13];
rz(6.249937787901595) q[3];
rz(2.7769000703822235) q[23];
rz(3.2597479550185295) q[22];
rz(1.7462295234117569) q[16];
rz(4.495043581305101) q[20];
rz(4.143832167812938) q[5];
rz(5.310283194144827) q[18];
rz(1.7046024580284937) q[9];
rz(1.4126286497579916) q[21];
rz(2.5238790868506724) q[17];
rz(3.4854421888107403) q[6];
rz(2.0708232540936526) q[24];
rz(0.5541739619721628) q[26];
rz(3.8138981079581358) q[15];
rz(0.04045491345928574) q[27];
cx q[0], q[10];
rz(0.43499260874548734) q[8];
rz(6.198573518890069) q[14];
rz(0.6207620699885081) q[7];
rz(2.382498715760971) q[25];
cx q[4], q[1];
rz(2.0251282461568194) q[11];
cx q[29], q[28];
rz(0.3487375243575893) q[28];
rz(5.750519135397606) q[11];
cx q[10], q[1];
rz(3.0172166187377423) q[22];
rz(5.932079432579371) q[18];
rz(3.321240942933411) q[24];
rz(0.26852055755081833) q[17];
rz(5.8530135482217265) q[12];
rz(5.0076395633365705) q[6];
rz(5.724920611062626) q[2];
rz(1.2564476730914327) q[27];
rz(3.9681940488846794) q[14];
rz(2.0273849442801577) q[16];
rz(3.9703670696134483) q[26];
rz(2.9136994038290145) q[8];
cx q[25], q[23];
rz(4.077855614419318) q[7];
rz(1.3136477450977082) q[29];
rz(2.9701712922892445) q[21];
rz(3.5162766540015578) q[20];
rz(1.9585249706704382) q[4];
rz(5.1565926125526165) q[19];
rz(2.754224836847127) q[3];
rz(5.838680106317005) q[13];
rz(1.0594539447948743) q[0];
rz(2.688468440171629) q[9];
rz(0.9082233636445346) q[15];
rz(3.3307655273363737) q[5];
rz(3.4619021368461884) q[25];
cx q[13], q[17];
rz(2.4936717924920475) q[3];
rz(1.4939986654132957) q[26];
cx q[27], q[9];
rz(4.5822335038476165) q[29];
rz(6.032290117273664) q[15];
cx q[5], q[12];
rz(0.9565487911901744) q[4];
rz(1.9507628374744677) q[19];
rz(1.7441201346606496) q[6];
rz(0.6345466035063031) q[10];
rz(3.8037359303386644) q[2];
rz(4.782765077810469) q[8];
rz(2.5919374166954) q[11];
rz(0.6777248138679903) q[18];
rz(4.52783207232011) q[1];
cx q[28], q[14];
rz(2.2711727164042137) q[22];
rz(4.566388461655479) q[23];
rz(5.8474551315774725) q[16];
rz(0.8487463433455149) q[21];
rz(1.8417339633425105) q[7];
cx q[20], q[0];
rz(6.250792466487899) q[24];
rz(2.9659794785893223) q[18];
rz(1.5837440654471007) q[5];
rz(4.838430209287504) q[23];
rz(6.216646219336686) q[13];
rz(0.7837688596014342) q[8];
rz(2.2926338042999146) q[27];
rz(4.859481829806054) q[15];
rz(1.9266912522185815) q[14];
rz(1.2261913461800986) q[19];
rz(2.749424028389375) q[0];
rz(0.1813922989951454) q[10];
rz(0.7089788610955112) q[11];
cx q[2], q[7];
rz(4.936204611146934) q[29];
rz(4.147416356364565) q[28];
rz(5.5527814276379415) q[16];
rz(3.8953277584301214) q[3];
rz(3.367432955430186) q[25];
cx q[22], q[4];
cx q[12], q[20];
rz(0.16767690953410042) q[26];
rz(1.4070163070461867) q[1];
rz(5.622635314302774) q[6];
rz(1.3521674034283109) q[21];
rz(4.1279337220283505) q[24];
cx q[17], q[9];
rz(0.9901643759554701) q[28];
cx q[10], q[18];
rz(2.8540931988015354) q[8];
rz(1.4373901790558181) q[21];
rz(4.18782520803186) q[6];
rz(1.8931249268136103) q[9];
rz(2.3501566310250808) q[19];
rz(3.887503452555778) q[2];
rz(4.30339966014813) q[3];
rz(3.230251942105255) q[4];
rz(4.154257115916114) q[17];
rz(0.6171839570153487) q[22];
rz(0.9118528856034731) q[7];
cx q[27], q[26];
cx q[24], q[25];
cx q[1], q[13];
rz(5.517600159808492) q[0];
rz(4.0413638850446105) q[12];
rz(4.970706071966313) q[23];
rz(5.967525912143275) q[5];
rz(5.554717097453094) q[11];
rz(5.460625788247303) q[29];
cx q[20], q[15];
rz(4.102533434882475) q[16];
rz(3.116895491906773) q[14];
rz(4.212565794459554) q[10];
rz(5.936334497157916) q[7];
rz(0.9502880016047965) q[16];
rz(3.622458507223486) q[24];
rz(2.3865481553860506) q[4];
rz(3.1040099121971894) q[9];
rz(1.9193069590632048) q[0];
rz(2.1173649968510846) q[3];
cx q[21], q[19];
rz(1.6516753074767563) q[29];
cx q[28], q[25];
rz(2.8066553544006285) q[22];
rz(2.7935382239919186) q[15];
cx q[8], q[1];
rz(2.283391636521205) q[11];
rz(5.287045573192475) q[27];
rz(0.6118222882217228) q[20];
rz(2.751610903109315) q[2];
rz(5.461891038895365) q[14];
rz(1.577381677151889) q[12];
rz(5.141794463031164) q[18];
rz(1.283679626164856) q[5];
rz(3.980365896512398) q[6];
rz(4.852046686046746) q[13];
rz(2.0512057772602237) q[17];
cx q[23], q[26];
rz(4.036647812995249) q[8];
rz(4.267910407339629) q[22];
cx q[9], q[26];
rz(2.2876036214380058) q[29];
cx q[5], q[24];
rz(0.3544617944245393) q[12];
rz(5.336426248903249) q[1];
rz(1.7689000716931633) q[10];
rz(1.5087916741867722) q[23];
rz(1.0549539029291812) q[19];
rz(4.450438135951928) q[17];
cx q[7], q[27];
rz(3.8011986754483327) q[3];
rz(4.611792646087539) q[25];
rz(4.4876743807351716) q[4];
rz(0.1811513050416111) q[11];
cx q[21], q[18];
cx q[15], q[28];
rz(0.45018786514399584) q[16];
cx q[13], q[20];
rz(4.937344757632156) q[14];
rz(4.436296478614809) q[2];
rz(4.217236034411492) q[0];
rz(0.8380070095308786) q[6];
rz(1.3409246732115958) q[22];
rz(5.232334138093982) q[18];
cx q[12], q[15];
rz(4.969073347134487) q[6];
cx q[16], q[0];
rz(5.934117854769693) q[11];
rz(3.4231221660205793) q[13];
cx q[2], q[10];
rz(1.0301263569046508) q[24];
rz(0.8751642705309731) q[27];
rz(5.637007667617417) q[9];
rz(5.878411994963062) q[23];
rz(4.677980553586981) q[20];
rz(4.3647662962713545) q[17];
rz(1.1157007696344425) q[3];
cx q[8], q[21];
rz(4.594823666962761) q[1];
rz(0.9601317855392103) q[26];
cx q[19], q[28];
cx q[4], q[14];
rz(3.278463627483127) q[25];
rz(1.20290304148761) q[5];
rz(1.5314689539167274) q[7];
rz(4.457794878181527) q[29];
cx q[29], q[7];
cx q[14], q[5];
rz(2.0692472111886766) q[17];
cx q[18], q[11];
rz(4.152292193448228) q[2];
rz(2.4931596443793973) q[28];
rz(1.261262581221574) q[1];
rz(5.1693853659048905) q[26];
rz(5.837963540545466) q[16];
rz(3.827830319427283) q[6];
rz(3.3174774318420757) q[9];
rz(1.8540914525229086) q[20];
rz(5.645977134080727) q[12];
rz(5.12776801381807) q[0];
rz(5.731398573255071) q[13];
rz(0.3744325654615625) q[8];
rz(5.944000706782042) q[4];
rz(2.9982708301746714) q[22];
rz(5.037690702026123) q[10];
rz(5.5526901787708) q[27];
cx q[24], q[19];
rz(2.7520325502019616) q[25];
rz(4.617324198817561) q[21];
rz(2.1795518135120338) q[23];
cx q[15], q[3];
cx q[2], q[4];
cx q[13], q[27];
rz(4.031221718691709) q[8];
rz(0.23196694256892428) q[1];
rz(4.501603683428151) q[19];
rz(6.048094388327081) q[17];
rz(3.7201745634875887) q[21];
cx q[6], q[3];
rz(3.3968550923598633) q[26];
rz(3.611715342820004) q[14];
rz(1.3680331922934668) q[23];
rz(5.433133726347223) q[25];
rz(1.0080981087533964) q[10];
rz(1.8180673548772064) q[16];
rz(1.046314590694671) q[15];
rz(3.2519778744772116) q[29];
rz(1.9305423879884454) q[24];
cx q[5], q[12];
rz(3.223522150295021) q[0];
rz(2.9325283246330525) q[18];
rz(3.8798688434657604) q[11];
rz(0.9922087418970833) q[7];
rz(2.4022902588223176) q[28];
rz(3.085811666419186) q[22];
cx q[20], q[9];
cx q[24], q[2];
rz(2.906226433352044) q[27];
cx q[21], q[23];
cx q[13], q[1];
rz(2.617421132520387) q[11];
rz(4.6423235317274205) q[14];
rz(2.5344744223698705) q[17];
rz(3.0713124016066584) q[5];
rz(3.430815907738405) q[28];
rz(2.8471134636975135) q[22];
cx q[20], q[29];
rz(6.025030877073306) q[6];
cx q[16], q[9];
rz(0.5687177424826355) q[8];
rz(0.21886553881516785) q[12];
rz(1.6679714693082182) q[15];
rz(5.819540321625741) q[10];
rz(3.3670810098134094) q[3];
rz(5.04781694759179) q[25];
rz(0.05021584244100103) q[19];
rz(5.584509991150716) q[18];
rz(2.645731489404783) q[26];
cx q[4], q[0];
rz(3.608604830167743) q[7];
cx q[17], q[0];
rz(0.851825362504702) q[1];
rz(0.5161455674127109) q[5];
rz(5.725468932658574) q[14];
cx q[2], q[4];
rz(5.416639327938639) q[16];
rz(2.264400180200569) q[12];
rz(5.277486354792994) q[13];
cx q[19], q[26];
rz(0.39906440037371277) q[15];
cx q[7], q[22];
rz(0.34516846543714286) q[10];
rz(1.4648935069794473) q[28];
rz(3.5102339034812644) q[20];
rz(5.048886365362652) q[6];
rz(3.126260187938327) q[21];
rz(0.46128185279157213) q[24];
cx q[29], q[23];
cx q[9], q[25];
cx q[8], q[27];
rz(2.3573712960840973) q[11];
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
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
measure q[20] -> c[20];
measure q[21] -> c[21];
measure q[22] -> c[22];
measure q[23] -> c[23];
measure q[24] -> c[24];
measure q[25] -> c[25];
measure q[26] -> c[26];
measure q[27] -> c[27];
measure q[28] -> c[28];
measure q[29] -> c[29];
