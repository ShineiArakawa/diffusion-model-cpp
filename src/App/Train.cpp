#include <torch/torch.h>

#include <DiffusionModelC++/Config/Config.hpp>
#include <DiffusionModelC++/Diffusion/KarrasDiffusion.hpp>
#include <DiffusionModelC++/Model/Model.hpp>
#include <DiffusionModelC++/Trainer/Trainer.hpp>
#include <memory>

struct Arguments {
  std::string config;

  static Arguments parseArgs(int argc, char* argv[]) {
    Arguments args;

    bool toShowHelp = false;

    if (argc < 2) {
      toShowHelp = true;
    }

    for (int i = 1; i < argc; ++i) {
      std::string arg = std::string(argv[i]);

      if (arg == "-h") {
        toShowHelp = true;
        break;
      } else {
        args.config = std::string(arg);
      }
    }

    if (toShowHelp) {
      std::cout << "############################################### diffuion-model-C++ ##############################################\n";
      std::cout << "                                                                                                                 \n";
      std::cout << "A trainer program for diffusion models.                                                                          \n";
      std::cout << "                                                                                                                 \n";
      std::cout << "usege: ./train [Options] config_file                                                                             \n";
      std::cout << "                                                                                                                 \n";
      std::cout << "[Options]                                                                                                        \n";
      std::cout << "  General                                                                                                        \n";
      std::cout << "    -h                                                                  Show this help message                   \n";
      exit(EXIT_SUCCESS);
    }

    return args;
  }
};

int main(int argc, char* argv[]) {
  const Arguments args = Arguments::parseArgs(argc, argv);

  // Load config
  const auto config = dmcpp::config::Config::load(args.config);

  // Set seed
  torch::manual_seed(config.seed);

  // Diffusion model
  dmcpp::diffusion::KarrasDiffusion diffusion = dmcpp::getDiffusionModel(config);
  dmcpp::diffusion::KarrasDiffusion diffusionEMA = dmcpp::getDiffusionModel(config);

  // Trainer
  auto trainer = std::make_shared<dmcpp::trainer::Trainer>(config, diffusion, diffusionEMA);

  trainer->fit();

  return 0;
}
