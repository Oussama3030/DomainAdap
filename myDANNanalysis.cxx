#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/Logger.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/Core/TrackSelection.h"
#include "Common/Core/TrackSelectionDefaults.h"
#include "Common/DataModel/PIDResponse.h"
#include "Framework/ASoA.h"
#include "Framework/O2DatabasePDGPlugin.h"
#include "TLorentzVector.h"
#include <iostream>
#include <TFile.h>
#include <TTree.h>
#include <vector>
#include <Math/Vector4D.h>

#include "Tools/ML/MlResponse.h"
#include "Tools/ML/model.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::analysis;
using namespace o2::framework::expressions;
using namespace o2::ml;
using namespace o2::soa;

/
// // DANN
static constexpr double defaultCutsMl[1][1] = {{4.39496}};

// MLP 
// static constexpr double defaultCutsMl[1][1] = {{3.90218}};

struct applyDANNSelection {
  HistogramRegistry histos{"histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  Service<o2::framework::O2DatabasePDG> pdg;

  // Configuration parameters
  Configurable<double> ptCandMin{"ptCandMin", 0.0, "Lower bound of candidate pT"};
  Configurable<double> ptCandMax{"ptCandMax", 5, "Upper bound of candidate pT"};

  // ML Configurables
  Configurable<std::vector<double>> binsPtMl{"binsPtMl", std::vector<double>{0., 36}, "pT bin limits for ML application"};
  Configurable<std::vector<int>> cutDirMl{"cutDirMl", std::vector<int>{cuts_ml::CutSmaller}, "Whether to reject score values greater or smaller than the threshold"};
  Configurable<LabeledArray<double>> cutsMl{"cutsMl", {defaultCutsMl[0], 1, 1, {"pT bin 0"}, {"score"}}, "ML selections per pT bin"};
  Configurable<int8_t> nClassesMl{"nClassesMl", (int8_t)1, "Number of classes in ML model"};

  // ML Configurables for features
  Configurable<std::vector<int>> cutDirFeature{"cutDirFeature", std::vector<int>{cuts_ml::CutNot}, "Whether to reject score values greater or smaller than the threshold"};
  Configurable<int16_t> nClassesFt{"nClassesFt", (int16_t)128, "Number of classes in ML model"};

  // DANN
  Configurable<std::vector<std::string>> onnxFileNames_0{"onnxFileNames_0", std::vector<std::string>{"DANN_ClassClassifier_CB.onnx"}, "ONNX file names for each pT bin"};
  Configurable<std::vector<std::string>> onnxFeatureName{"onnxFeatureName", std::vector<std::string>{"DANN_FeatureExtractor_CB.onnx"}, "ONNX file names for each pT bin"};
  
  // // MLP
  // Configurable<std::vector<std::string>> onnxFileNames_0{"onnxFileNames_0", std::vector<std::string>{"MLP_Classifier_FT_CB.onnx"}, "ONNX file names for each pT bin"};
  // Configurable<std::vector<std::string>> onnxFeatureName{"onnxFeatureName", std::vector<std::string>{"MLP_FT_CB.onnx"}, "ONNX file names for each pT bin"};

  // Objects for ML inference
  std::vector<float> outputFeatures = {};
  std::vector<float> outputMl = {};

  std::vector<float> outputFeatures_1 = {};
  std::vector<float> outputMl_1 = {};

  o2::analysis::MlResponse<float> mlResponse;
  o2::analysis::MlResponse<float> mlResponseFeature;

  o2::analysis::MlResponse<float> mlResponse_1;
  o2::analysis::MlResponse<float> mlResponseFeature_1;

  // Count candidates for logging
  int nCandidates = 0;
  int nSelectedCandidates = 0;

  void init(InitContext const&)
  {
    // Create histograms
    const AxisSpec pAxis(500, 0, 10);
    const AxisSpec InvMassAxis(500, 0, 5);
    const AxisSpec TPCAxis(500, -2.5, 2.5);
    const AxisSpec TOFAxis(500, -2.5, 2.5);
    const AxisSpec MLScoreAxis(500, 0, 1);
    const AxisSpec RawMLScoreAxis(100, -100, 100);
    const AxisSpec Count(10, 0, 10);

    // Basic histograms
    histos.add("pT", "pT Distribution", kTH1F, {pAxis});
    histos.add("pT_selected", "pT Distribution (Selected)", kTH1F, {pAxis});

    histos.add("tpcNSigmaEl", "TPC NSigma Electron", kTH1F, {TPCAxis});
    histos.add("tpcNSigmaEl_selected", "TPC NSigma Electron (Selected)", kTH1F, {TPCAxis});

    histos.add("tofNSigmaEl", "TOF NSigma Electron", kTH1F, {TPCAxis});
    histos.add("tofNSigmaEl_selected", "TOF NSigma Electron (Selected)", kTH1F, {TPCAxis});

    histos.add("MLScore", "ML Score Distribution", kTH1F, {MLScoreAxis});
    histos.add("MLScore_signal", "ML Score (True e+/e-)", kTH1F, {MLScoreAxis});
    histos.add("MLScore_background", "ML Score (Background)", kTH1F, {MLScoreAxis});

    // 2D histograms
    histos.add("tpcVStofNSigmaEl", "TPC vs TOF NSigma Electron", kTH2F, {TPCAxis, TOFAxis});
    histos.add("pVSTPCSelected", "p vs TPC (Selected)", kTH2F, {pAxis, TPCAxis});

    histos.add("pTvsMLScore", "pT vs ML Score", kTH2F, {pAxis, MLScoreAxis});
    histos.add("pTvsMLScore_selected", "pT vs ML Score (Selected)", kTH2F, {pAxis, MLScoreAxis});

    // Prob Distributions
    histos.add("MLProb", "ML Prob", kTH1F, {MLScoreAxis});
    histos.add("MLProb_v1", "ML Prob v1", kTH1F, {MLScoreAxis});
    histos.add("MLProb_v2", "ML Prob v2", kTH1F, {MLScoreAxis});

    histos.add("RawMLScore", "Raw ML Score", kTH1F, {RawMLScoreAxis});

    histos.add("InvMassOpp", "Invariant Mass El Opp", kTH1F, {InvMassAxis});
    histos.add("InvMassNeg", "Invariant Mass El Neg", kTH1F, {InvMassAxis});
    histos.add("InvMassPos", "Invariant Mass El Pos", kTH1F, {InvMassAxis});

    histos.add("electroncount", "Electron Count", kTH1F, {pAxis});
    histos.add("positroncount", "Positron Count", kTH1F, {pAxis});

    histos.add("Collision", "Number of Collisions", kTH1F, {Count});

    // First track

    mlResponse.configure(binsPtMl, cutsMl, cutDirMl, nClassesMl);
    mlResponse.setModelPathsLocal(onnxFileNames_0);
    mlResponse.init();

    // Expand the vector to match the number of features
    if (cutDirFeature.value.size() == 1 && nClassesFt.value > 1) {
      int value = cutDirFeature.value[0];
      cutDirFeature.value = std::vector<int>(nClassesFt.value, value);
      LOGF(info, "Expanded cutDirFeature to %d elements all set to %d", nClassesFt.value, value);
    }

    mlResponseFeature.configure(binsPtMl, cutsMl, cutDirFeature, nClassesFt);
    mlResponseFeature.setModelPathsLocal(onnxFeatureName);
    mlResponseFeature.init();

    // Second track

    mlResponse_1.configure(binsPtMl, cutsMl, cutDirMl, nClassesMl);
    mlResponse_1.setModelPathsLocal(onnxFileNames_0);
    mlResponse_1.init();

    mlResponseFeature_1.configure(binsPtMl, cutsMl, cutDirFeature, nClassesFt);
    mlResponseFeature_1.setModelPathsLocal(onnxFeatureName);
    mlResponseFeature_1.init();

    LOGF(info, "ML model initialized successfully");
  }

  float particleMass(int pid)
  {
    auto mass = pdg->Mass(pid);
    return mass;
  }

  void process(aod::Collision const& collisions,
               //    aod::McParticles_001 const& mcparticles,
               soa::Join<aod::Tracks, aod::TrackSelection,
                         aod::TracksExtra_001, aod::TracksDCA,
                         aod::TrackSelectionExtension,
                         aod::pidTOFbeta, // aod::McTrackLabels,
                         aod::pidTPCFullPi, aod::pidTOFFullPi,
                         aod::pidTPCFullEl, aod::pidTOFFullEl,
                         aod::pidTPCFullPr, aod::pidTOFFullPr,
                         aod::pidTPCFullKa, aod::pidTOFFullKa,
                         aod::pidTPCFullMu, aod::pidTOFFullMu,
                         aod::pidTOFmass, aod::TOFSignal> const& tracks)
  {
    for (auto& [t0, t1] : combinations(CombinationsFullIndexPolicy(tracks, tracks))) { // combiantions(CombinationsFullIndexPolicy())

      // No same tuple index
      if (t0.globalIndex() == t1.globalIndex())
        continue;

      // Track 1
      if (std::abs(t0.eta()) > 0.8 || t0.pt() < 0.1)
        continue;
      if (t0.tpcNClsCrossedRows() < 70)
        continue;
      if (t0.itsClsSizeInLayer(0) == 0)
        continue;
      if (!t0.hasTPC())
        continue;
      if (!t0.hasTOF())
        continue;

      // Track 2
      if (std::abs(t1.eta()) > 0.8 || t1.pt() < 0.1)
        continue;
      if (t1.tpcNClsCrossedRows() < 70)
        continue;
      if (t1.itsClsSizeInLayer(0) == 0)
        continue;
      if (!t1.hasTPC())
        continue;
      if (!t1.hasTOF())
        continue;

      auto mEl = particleMass(11);

      // Check pT range
      auto candpT_0 = t0.p();
      auto candpT_1 = t1.p();
      if (candpT_0 < ptCandMin || candpT_0 > ptCandMax)
        continue;
      if (candpT_1 < ptCandMin || candpT_1 > ptCandMax)
        continue;

      // // Calculate features for ML input
      float TPCratioEl = (t0.tpcExpSignalDiffEl() / t0.tpcExpSignalEl(t0.tpcSignal()));
      float TOFratioEl = (t0.tofExpSignalDiffEl() / t0.tofExpSignalEl(t0.tofSignal()));
      float tofSignalLog = TPCratioEl * log(1 + t0.p());
      float tpcSignalExp = TPCratioEl * exp(-1 * t0.p());

      float TPCratioEl_1 = (t1.tpcExpSignalDiffEl() / t1.tpcExpSignalEl(t0.tpcSignal()));
      float TOFratioEl_1 = (t1.tofExpSignalDiffEl() / t1.tofExpSignalEl(t0.tofSignal()));
      float tofSignalLog_1 = TPCratioEl_1 * log(1 + t1.p());
      float tpcSignalExp_1 = TPCratioEl_1 * exp(-1 * t1.p());

      //   // Fill pre-selection histograms
      //   histos.fill(HIST("tpcNSigmaEl"), track.tpcNSigmaEl());
      //   histos.fill(HIST("tofNSigmaEl"), track.tofNSigmaEl());

      std::vector<float> inputFeaturesMl{
        TPCratioEl,
        t0.p(),
        TOFratioEl,
        t0.tofNSigmaEl(),
        t0.tpcNSigmaEl(),
        tofSignalLog,
        tpcSignalExp};

      std::vector<float> inputFeaturesMl_1{
        TPCratioEl_1,
        t1.p(),
        TOFratioEl_1,
        t1.tofNSigmaEl(),
        t1.tpcNSigmaEl(),
        tofSignalLog_1,
        tpcSignalExp_1};

      // Track 1

      std::vector<float> featureModelOutput = mlResponseFeature.getModelOutput(inputFeaturesMl, 0);
      bool isSelectedMl = mlResponse.isSelectedMl(featureModelOutput, t0.p(), outputMl);

      // Track 2

      std::vector<float> featureModelOutput_1 = mlResponseFeature_1.getModelOutput(inputFeaturesMl_1, 0);
      bool isSelectedMl_1 = mlResponse_1.isSelectedMl(featureModelOutput_1, t1.p(), outputMl_1);

      // MLP Threshold: 0.987811 -> 4.39
      // DANN Threshold: 2.33

      if (!outputMl_1.empty() && !outputMl.empty()) {

        // Sigmoid
        for (size_t i = 0; i < outputMl.size(); i++) {
          outputMl[i] = 1.0 / (1.0 + std::exp(-outputMl[i]));
        }
        float mlScore = outputMl[0];

        for (size_t i = 0; i < outputMl_1.size(); i++) {
          outputMl_1[i] = 1.0 / (1.0 + std::exp(-outputMl_1[i]));
        }
        float mlScore_1 = outputMl_1[0];

        histos.fill(HIST("MLScore"), mlScore);

        // histos.fill(HIST("MLScore"), mlScore);
        if (isSelectedMl) {
          //   LOGF(info, "Found with Global indices: %d and %d", t0.globalIndex(), t1.globalIndex());
          if (t0.sign() == 1) {
            histos.fill(HIST("MLScore_signal"), mlScore);
          }

          if (t0.sign() == -1) {
            histos.fill(HIST("MLScore_background"), mlScore);
          }

          histos.fill(HIST("pVSTPCSelected"), t0.p(), TPCratioEl);
        }

        // Check for this outside of the if statement whether raw outputs or sigmoid
        if (isSelectedMl && isSelectedMl_1) {

          if (t0.sign() == 1 && t1.sign() == 1) {

            // LOGF(info, "Found a e+e+ pair with global indices: %d and %d", t0.globalIndex(), t1.globalIndex());
            // // LOGF(info, "ML score 0: %.3f", outputMl[0]);
            // // LOGF(info, "ML score 1: %.3f", outputMl_1[0]);
            // LOGF(info, "with Global indices: %d and %d", t0.globalIndex(), t1.globalIndex());
            // LOGF(info, "with Collision ID: %d and %d", t0.collisionId(), t1.collisionId());

            TLorentzVector v0;
            TLorentzVector v1;

            v0.SetXYZM(t0.px(), t0.py(), t0.pz(), mEl);
            v1.SetXYZM(t1.px(), t1.py(), t1.pz(), mEl);

            TLorentzVector v = v0 + v1;

            histos.fill(HIST("InvMassOpp"), v.M());
          }
          if (t0.sign() == -1 && t1.sign() == -1) {

            // LOGF(info, "Found a e-e- pair");
            // // LOGF(info, "ML score 0: %.3f", outputMl[0]);
            // // LOGF(info, "ML score 1: %.3f", outputMl_1[0]);
            // LOGF(info, "with Global indices: %d and %d", t0.globalIndex(), t1.globalIndex());
            // LOGF(info, "with Collision ID: %d and %d", t0.collisionId(), t1.collisionId());

            TLorentzVector u0;
            TLorentzVector u1;

            u0.SetXYZM(t0.px(), t0.py(), t0.pz(), mEl);
            u0.SetXYZM(t1.px(), t1.py(), t1.pz(), mEl);

            TLorentzVector u = u0 + u1;

            histos.fill(HIST("InvMassNeg"), u.M());

            // fill -- histogram
          }
          if (t0.sign() != t1.sign()) {
            // LOGF(info, "Found a e+e- pair");
            // // LOGF(info, "ML score 0: %.3f", outputMl[0]);
            // // LOGF(info, "ML score 1: %.3f", outputMl_1[0]);
            // LOGF(info, "with Global indices: %d and %d", t0.globalIndex(), t1.globalIndex());
            // LOGF(info, "with Collision ID: %d and %d", t0.collisionId(), t1.collisionId());

            TLorentzVector p0;
            TLorentzVector p1;

            p0.SetXYZM(t0.px(), t0.py(), t0.pz(), mEl);
            p1.SetXYZM(t1.px(), t1.py(), t1.pz(), mEl);

            TLorentzVector p = p0 + p1;

            histos.fill(HIST("InvMassPos"), p.M());

            // fill +- histogram
          }

          //   LOGF(info, "ML score Selected: %.3f", outputMl[0]);
        }
      }
    }
  }
};

WorkflowSpec
  defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<applyDANNSelection>(cfgc)};
  // adaptAnalysisTask<applyDANNSelection>(cfgc)};
}
