# Starforge — 3D real-time strategy. Apple Silicon / Metal, no external deps.
APP      := starforge
BUILD    := build
SRC      := src

CXX      := clang++
WARN     := -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
CXXFLAGS := -std=c++20 -O2 -arch arm64 -fvisibility=hidden -I$(SRC) $(WARN)
MMFLAGS  := $(CXXFLAGS) -fobjc-arc
LDFLAGS  := -arch arm64 -framework Cocoa -framework Metal -framework QuartzCore \
            -framework CoreText -framework CoreGraphics -framework ImageIO

CPP_SRCS := $(SRC)/sim/Terrain.cpp $(SRC)/sim/Nav.cpp $(SRC)/sim/Game.cpp $(SRC)/gfx/MeshGen.cpp \
            $(SRC)/ai/Brain.cpp $(SRC)/ai/Strategy.cpp $(SRC)/ai/Commander.cpp
MM_SRCS  := $(SRC)/gfx/Renderer.mm $(SRC)/ai/InfluenceMapGPU.mm $(SRC)/app/main.mm
OBJS     := $(CPP_SRCS:$(SRC)/%.cpp=$(BUILD)/%.o) $(MM_SRCS:$(SRC)/%.mm=$(BUILD)/%.o)
DEPS     := $(OBJS:.o=.d)

.PHONY: all run clean bench aieval

all: $(APP)

$(APP): $(OBJS)
	@echo "  LINK  $@"
	@$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(BUILD)/%.o: $(SRC)/%.cpp
	@mkdir -p $(dir $@)
	@echo "  CXX   $<"
	@$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: $(SRC)/%.mm
	@mkdir -p $(dir $@)
	@echo "  OBJCXX $<"
	@$(CXX) $(MMFLAGS) -MMD -MP -c $< -o $@

run: $(APP)
	./$(APP)

bench: $(APP)
	./$(APP) --bench 300 --no-help

# Headless AI evaluation: adaptive AI vs scripted opponents of distinct styles.
aieval:
	@$(CXX) $(CXXFLAGS) tools/ai_eval.cpp $(SRC)/sim/Game.cpp $(SRC)/sim/Nav.cpp \
	  $(SRC)/sim/Terrain.cpp $(SRC)/ai/Brain.cpp $(SRC)/ai/Strategy.cpp \
	  $(SRC)/ai/Commander.cpp -o $(BUILD)/ai_eval
	@$(BUILD)/ai_eval $(GAMES) $(SECS)

GAMES ?= 8
SECS  ?= 600

clean:
	@rm -rf $(BUILD) $(APP)

-include $(DEPS)
