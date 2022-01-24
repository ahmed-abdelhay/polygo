#define APPLICATION_CODE
#define POLYGO_IMPLEMENTATION

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <charconv>
#include <chrono>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#if defined(POLYGO_IMPLEMENTATION)
#include <GL/gl3w.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>
#endif

template <typename F> struct privDefer {
  F f;
  privDefer(F f) : f(f) {}
  ~privDefer() { f(); }
};
template <typename F> privDefer<F> defer_func(F f) { return privDefer<F>(f); }
#define DEFER_1(x, y) x##y
#define DEFER_2(x, y) DEFER_1(x, y)
#define DEFER_3(x) DEFER_2(x, __COUNTER__)
#define defer(code) auto DEFER_3(_defer_) = defer_func([&]() { code; })

//------------------Math----------------------------//
constexpr double PI = 3.14159265358979323846264338327950288;
double Deg2Rad(double v) { return v * (PI / 180); }
double Rad2Deg(double v) { return v * (180 / PI); }

union Vec2f {
  union {
    struct {
      float x, y;
    };
    float data[2];
  };

  Vec2f operator+(const Vec2f &b) const { return Vec2f{x + b.x, y + b.y}; }
  Vec2f operator-(const Vec2f &b) const { return Vec2f{x - b.x, y - b.y}; }
  Vec2f operator*(const Vec2f &b) const { return Vec2f{x * b.x, y * b.y}; }
  Vec2f operator*(float s) const { return Vec2f{x * s, y * s}; }
  bool operator==(const Vec2f &b) const { return x == b.x && y == b.y; }
  bool operator!=(const Vec2f &b) const { return x != b.x || y != b.y; }
};

struct Vec3f {
  union {
    struct {
      float x, y, z;
    };
    float data[3];
  };
  Vec3f operator+(const Vec3f &b) const {
    return Vec3f{x + b.x, y + b.y, z + b.z};
  }
  Vec3f operator-(const Vec3f &b) const {
    return Vec3f{x - b.x, y - b.y, z - b.z};
  }
  Vec3f operator*(const Vec3f &b) const {
    return Vec3f{x * b.x, y * b.y, z * b.z};
  }
  Vec3f operator*(float s) const { return Vec3f{x * s, y * s, z * s}; }
  bool operator==(const Vec3f &b) const {
    return x == b.x && y == b.y && z == b.z;
  }
  bool operator!=(const Vec3f &b) const {
    return x != b.x || y != b.y || z != b.z;
  }
  size_t Hash() const {
    const int HASH_SIZE = 200;
    const float L = 0.2f;

    const int ix = (unsigned int)((x + 2.f) / L);
    const int iy = (unsigned int)((y + 2.f) / L);
    const int iz = (unsigned int)((z + 2.f) / L);
    return (unsigned int)((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) %
           HASH_SIZE;
  }
};

struct Vec3fHash {
  size_t operator()(const Vec3f &v) const { return v.Hash(); }
};

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
Vec3f Lerp(float isolevel, Vec3f p1, Vec3f p2, float valp1, float valp2);

// the matrices are column major.
struct Mat3 {
  union {
    float elements[3][3];
    float data[9];
  };
  Vec3f operator*(const Vec3f &v) const;
};

struct Mat4 {
  union {
    float elements[4][4];
    float data[16];
  };
  Mat4 operator*(const Mat4 &right) const;
};

struct BBox {
  Vec3f min{FLT_MAX, FLT_MAX, FLT_MAX};
  Vec3f max{-FLT_MAX, -FLT_MAX, -FLT_MAX};
};

bool IsBBoxValid(const BBox &box);
Vec3f CalculateBBoxCenter(const BBox &box);
void Merge(BBox &box, const Vec3f &v);
BBox Merge(const BBox &a, const BBox &b);

Vec3f CrossProduct(const Vec3f &a, const Vec3f &b);

float DotProduct(const Vec3f &a, const Vec3f &b);
float DotProduct(const Vec2f &a, const Vec2f &b);

float Length(const Vec3f &v);
float Length(const Vec2f &v);

void Normalise(Vec3f &v);
void Normalise(Vec2f &v);

Vec3f Normalised(const Vec3f &v);
Vec2f Normalised(const Vec2f &v);

Mat4 Identity();
Mat4 Translate(Mat4 m, const Vec3f &translation);
Mat4 Rotate(const Mat4 &m, double angle, const Vec3f &v);
Mat4 Perspective(double fovy, double aspect, double zNear, double zFar);
Mat4 LookAt(const Vec3f &eye, const Vec3f &center, const Vec3f &up);

static size_t GenerateID() {
  static size_t id = 0;
  return id++;
}

//--------------------------------File IO---------------------//
template <typename T = uint8_t> struct Buffer {
  T *data = nullptr;
  size_t size = 0;
  size_t cursor = 0;

  T Peek() const {
    assert(cursor < size);
    return data[cursor];
  }
};
using RawBuffer = Buffer<uint8_t>;
template <typename T> void AppendData(const T &data, RawBuffer &buffer) {
  for (uint8_t i = 0; i < sizeof(T); i++) {
    buffer.data[buffer.cursor++] = ((uint8_t *)&data)[i];
  }
}

// File names are ASCII encoded.
bool ReadFile(const char *fileName, std::vector<uint8_t> &data);
bool WriteFile(const char *fileName, const uint8_t *data, size_t size);
//------------------------------------------------------------//

//-----------------------Time  -------------------------------//
#define TIME_BLOCK(BlockName)                                                  \
  StopWatch _t;                                                                \
  _t.Start();                                                                  \
  defer({                                                                      \
    _t.Stop();                                                                 \
    printf("Time spent in (%s): %f seconds.\n", BlockName,                     \
           _t.ElapsedSeconds());                                               \
  });

struct StopWatch {
  std::chrono::high_resolution_clock::time_point start, end;

  void Start() { start = std::chrono::high_resolution_clock::now(); }
  void Stop() { end = std::chrono::high_resolution_clock::now(); }
  double ElapsedSeconds() const {
    return std::chrono::duration_cast<std::chrono::seconds>(end - start)
        .count();
  }
};
//------------------------------------------------------------//

//------------------------------------------------------------//
struct Color {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 255;
};

struct Image3D {
  float *data = nullptr;
  size_t size[3] = {};
  float spacing[3] = {};
  float origin[3] = {};

  inline size_t LinearIndex(size_t x, size_t y, size_t z) const {
    return x + y * size[0] + z * size[0] * size[1];
  }
  inline float &At(size_t x, size_t y, size_t z) {
    return data[LinearIndex(x, y, z)];
  }
  inline const float &At(size_t x, size_t y, size_t z) const {
    return data[LinearIndex(x, y, z)];
  }
};

struct Mesh {
  struct Triangle {
    uint32_t idx[3];
  };
  std::vector<Vec3f> vertices;
  std::vector<Triangle> faces;
  std::string name;
  Color color;
  size_t id;
  bool visible = true;
};

struct Connectivity {
  struct VertexNeighbours {
    std::set<size_t> adjacentFaces;
  };
  std::vector<VertexNeighbours> pointCells;
};
BBox CalculateBBox(const Mesh &mesh);
std::vector<Vec3f> CalculateFacesNormals(const Mesh &mesh);
bool WriteStl(const Mesh &mesh, const char *fileName);
Connectivity BuildConnectivity(const Mesh &mesh);
std::vector<Vec3f> CalculateVertexNormals(const Mesh &mesh,
                                          const Connectivity &connectivity);

Color GenerateColor();

// Graphics
enum class ShaderType { GEOMETRY, FRAGMENT, VERTEX };

struct Program {
  bool valid = false;

  int32_t id = -1;
  std::string geometryShader;
  std::string fragmentShader;
  std::string vertexShader;
};

struct RenderBuffer {
  size_t width = 0;
  size_t height = 0;
  int32_t frameBufferId = -1;
  int32_t renderBufferId = -1;
  int32_t textureId = -1; // id of the texture used to store the 3d render
                          // pipeline of the 3D view.
};

struct MeshRenderInfo {
  int32_t vertexBufferObject = -1;
  int32_t vertexBufferId = -1;
  int32_t elementBufferId = -1;
  size_t facesCount = 0;
  size_t verticesCount = 0;
  BBox box;
  size_t id;
};

RenderBuffer CreateRenderBuffer(size_t width, size_t height);
void ReizeRenderBuffer(RenderBuffer &buffer, size_t width, size_t heigth);
void ClearRenderBuffer(const RenderBuffer &buffer, Color c);

// geometry shader pointers can be null, in that case it won't be added to the
// output program.
Program CreateProgram(const char *geometryShader, const char *vertexShader,
                      const char *fragmentShader);

// returns -1 on failure and the shader id on success.
int32_t CompileShader(const char *shader, ShaderType type);

void SetProgramUniformV3f(const Program &program, const char *name,
                          const float data[3]);
void SetProgramUniformM4x4f(const Program &program, const char *name,
                            const float data[16]);

uint32_t GenerateTexture();
void UpdateTexture(uint32_t textureId, size_t width, size_t height,
                   Color *rgbaData);

MeshRenderInfo CreateMeshRenderInfo(Mesh &mesh);
void RenderMesh(const RenderBuffer &buffer, const Program &program,
                const MeshRenderInfo &info);

// 3D Camera
struct Camera {
  static constexpr float DEFAULT_NEAR_CLIP = 0.005f;
  static constexpr float DEFAULT_FAR_CLIP = 20.0f;
  static constexpr float DEFAULT_FOV = 45.0f;

  Mat4 viewMatrix = Identity();
  float lengthScale = 1.0f;
  Vec3f center = {0};
  float fov = DEFAULT_FOV;
  float nearClipRatio = DEFAULT_NEAR_CLIP;
  float farClipRatio = DEFAULT_FAR_CLIP;

  Mat4 GetViewMatrix() const;
  Mat4 GetProjectionMatrix(size_t width, size_t height) const;
  void GetFrame(Vec3f &look, Vec3f &up, Vec3f &right) const;
  void FitBBox(const BBox &box);
  void Zoom(float amount);
  void Rotate(Vec2f start, Vec2f end);
  void Translate(Vec2f delta);
};

// Expression parser
enum class ExpressionParseErrorType { OK, PARSE_ERROR, MULTIPLE_VARIABLES };

struct Expression {
  enum class NodeType {
    ADD,
    SUB,
    MULTIPLY,
    POWER,
    DIVIDE,
    SIN,
    COS,
    TAN,
    VARIABLE,
    CONSTANAT
  };

  enum class VariableType { X, Y, Z };

  struct Node {
    NodeType type;
    int32_t children[2] = {-1, -1};
    VariableType variable;
    double constant;
  };

  std::vector<Node> nodes;
  size_t rootNodeId = 0;

  double Evaluate(int32_t nodeId, double x, double y, double z) const {
    const Node &node = nodes[nodeId];
    switch (node.type) {
    case NodeType::ADD: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      const double v1 = Evaluate(node.children[1], x, y, z);
      return v0 + v1;
    } break;
    case NodeType::SUB: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      const double v1 = Evaluate(node.children[1], x, y, z);
      return v0 - v1;
    } break;
    case NodeType::DIVIDE: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      const double v1 = Evaluate(node.children[1], x, y, z);
      return v0 / v1;
    } break;
    case NodeType::MULTIPLY: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      const double v1 = Evaluate(node.children[1], x, y, z);
      return v0 * v1;
    } break;
    case NodeType::POWER: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      const double v1 = Evaluate(node.children[1], x, y, z);
      return pow(v0, v1);
    } break;
    case NodeType::SIN: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      return sin(v0);
    } break;
    case NodeType::COS: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      return cos(v0);
    } break;
    case NodeType::TAN: {
      const double v0 = Evaluate(node.children[0], x, y, z);
      return tan(v0);
    } break;
    case NodeType::VARIABLE: {
      switch (node.variable) {
      case VariableType::X:
        return x;
      case VariableType::Y:
        return y;
      case VariableType::Z:
        return z;
      }
    } break;
    case NodeType::CONSTANAT: {
      return node.constant;
    } break;
    }
    assert(false);
    return 0;
  }

  double Evaluate(double x, double y, double z) const {
    assert(!nodes.empty());
    return Evaluate(rootNodeId, x, y, z);
  }
};

ExpressionParseErrorType GenerateExpression(const char *text,
                                            Expression &result);

// Polygonisation
using FunctionType = std::function<float(float, float, float)>;
Mesh Polygonise(const FunctionType &expr, float isolevel, const float min[3],
                const float max[3], const float sampleDistance[3]);

#ifdef POLYGO_IMPLEMENTATION
bool ReadFile(const char *fileName, std::vector<uint8_t> &data) {
  FILE *fp = fopen(fileName, "r");
  if (!fp) {
    return false;
  }
  fseek(fp, 0, SEEK_END);
  const int64_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  data.resize(size);
  fread(data.data(), 1, size, fp);
  fclose(fp);
  return true;
}

bool WriteFile(const char *fileName, const uint8_t *data, size_t size) {
  FILE *fp = fopen(fileName, "w");
  if (fp) {
    fwrite(data, 1, size, fp);
    fclose(fp);
    return true;
  }
  return false;
}

bool WriteStl(const Mesh &mesh, const char *fileName) {
  const uint32_t facesCount = mesh.faces.size();

  std::vector<uint8_t> data;
  data.resize(80 + sizeof(uint32_t) +
              facesCount * (sizeof(uint16_t) + 12 * sizeof(float)));

  RawBuffer buffer;
  buffer.data = data.data();
  buffer.size = data.size();

  AppendData(facesCount, buffer);
  for (size_t i = 0; i < facesCount; ++i) {
    const Vec3f &v0 = mesh.vertices[mesh.faces[i].idx[0]];
    const Vec3f &v1 = mesh.vertices[mesh.faces[i].idx[1]];
    const Vec3f &v2 = mesh.vertices[mesh.faces[i].idx[2]];
    assert(!(isnan(v0.x) || isnan(v0.y) || isnan(v0.z)));
    assert(!(isnan(v1.x) || isnan(v1.y) || isnan(v1.z)));
    assert(!(isnan(v2.x) || isnan(v2.y) || isnan(v2.z)));

    const Vec3f dir0 = v1 - v0;
    const Vec3f dir1 = v2 - v0;
    Vec3f n = CrossProduct(dir0, dir1);
    Normalise(n);
    assert(!(isnan(n.x) || isnan(n.y) || isnan(n.z)));

    AppendData(n, buffer);
    AppendData(v0, buffer);
    AppendData(v1, buffer);
    AppendData(v2, buffer);
    AppendData(uint16_t(0), buffer);
  }
  return WriteFile(fileName, buffer.data, buffer.size);
}

BBox CalculateBBox(const Mesh &mesh) {
  BBox result;
  for (const Vec3f v : mesh.vertices) {
    Merge(result, v);
  }
  return result;
}

Vec3f Lerp(float isolevel, Vec3f p1, Vec3f p2, float valp1, float valp2) {
  constexpr float EPS = 0.00001;
  if (fabs(isolevel - valp1) < EPS) {
    return (p1);
  }
  if (fabs(isolevel - valp2) < EPS) {
    return (p2);
  }
  if (fabs(valp1 - valp2) < EPS) {
    return (p1);
  }
  const float mu = (isolevel - valp1) / (valp2 - valp1);
  Vec3f p = {};
  p.x = p1.x + mu * (p2.x - p1.x);
  p.y = p1.y + mu * (p2.y - p1.y);
  p.z = p1.z + mu * (p2.z - p1.z);
  return p;
}
Connectivity BuildConnectivity(const Mesh &mesh) {
  Connectivity c;
  const size_t pointsCount = mesh.vertices.size();
  c.pointCells.resize(pointsCount);
  const size_t facesCount = mesh.faces.size();
  for (size_t i = 0; i < facesCount; ++i) {
    const Mesh::Triangle &f = mesh.faces[i];
    c.pointCells[f.idx[0]].adjacentFaces.insert(i);
    c.pointCells[f.idx[1]].adjacentFaces.insert(i);
    c.pointCells[f.idx[2]].adjacentFaces.insert(i);
  }
  return c;
}

std::vector<Vec3f> CalculateVertexNormals(const Mesh &mesh,
                                          const Connectivity &connectivity) {
  const std::vector<Vec3f> faceNormals = CalculateFacesNormals(mesh);
  const size_t verticesCount = mesh.vertices.size();
  std::vector<Vec3f> normals(verticesCount);
  for (size_t i = 0; i < verticesCount; ++i) {
    normals[i] = Vec3f{0.0, 0.0, 0.0};
    for (const size_t faceIdx : connectivity.pointCells[i].adjacentFaces) {
      normals[i] = normals[i] + faceNormals[faceIdx];
    }
    const size_t adjecantFacesCount =
        connectivity.pointCells[i].adjacentFaces.size();
    normals[i] = normals[i] * (1.0 / adjecantFacesCount);
    Normalise(normals[i]);
  }
  return normals;
}

std::vector<Vec3f> CalculateFacesNormals(const Mesh &mesh) {
  std::vector<Vec3f> faceNormals;
  faceNormals.reserve(mesh.faces.size());
  for (const Mesh::Triangle &t : mesh.faces) {
    const Vec3f v0 = mesh.vertices[t.idx[0]];
    const Vec3f v1 = mesh.vertices[t.idx[1]];
    const Vec3f v2 = mesh.vertices[t.idx[2]];
    faceNormals.push_back(Normalised(CrossProduct(v1 - v0, v2 - v0)));
  }
  return faceNormals;
}

float DotProduct(const Vec3f &a, const Vec3f &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

float DotProduct(const Vec2f &a, const Vec2f &b) {
  return a.x * b.x + a.y * b.y;
}

Vec3f Mat3::operator*(const Vec3f &v) const {
  const Vec3f v0{elements[0][0], elements[0][1], elements[0][2]};
  const Vec3f v1{elements[1][0], elements[1][1], elements[1][2]};
  const Vec3f v2{elements[2][0], elements[2][1], elements[2][2]};
  return Vec3f{DotProduct(v0, v), DotProduct(v1, v), DotProduct(v2, v)};
}

Mat4 Mat4::operator*(const Mat4 &b) const {
  Mat4 r = {};
  for (int i = 0; i < 4; ++i) {
    r.elements[0][i] =
        elements[0][i] * b.elements[0][0] + elements[1][i] * b.elements[0][1] +
        elements[2][i] * b.elements[0][2] + elements[3][i] * b.elements[0][3];
    r.elements[1][i] =
        elements[0][i] * b.elements[1][0] + elements[1][i] * b.elements[1][1] +
        elements[2][i] * b.elements[1][2] + elements[3][i] * b.elements[1][3];
    r.elements[2][i] =
        elements[0][i] * b.elements[2][0] + elements[1][i] * b.elements[2][1] +
        elements[2][i] * b.elements[2][2] + elements[3][i] * b.elements[2][3];
    r.elements[3][i] =
        elements[0][i] * b.elements[3][0] + elements[1][i] * b.elements[3][1] +
        elements[2][i] * b.elements[3][2] + elements[3][i] * b.elements[3][3];
  }
  return r;
}

Vec3f CrossProduct(const Vec3f &a, const Vec3f &b) {
  return Vec3f{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x};
}

float Length(const Vec2f &v) { return sqrt(v.x * v.x + v.y * v.y); }

float Length(const Vec3f &v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

void Normalise(Vec2f &v) {
  const double length = Length(v);
  v.x /= length;
  v.y /= length;
}

void Normalise(Vec3f &v) {
  const double length = Length(v);
  v.x /= length;
  v.y /= length;
  v.z /= length;
}

Vec3f Normalised(const Vec3f &v) {
  Vec3f result = v;
  Normalise(result);
  return result;
}

Vec2f Normalised(const Vec2f &v) {
  Vec2f result = v;
  Normalise(result);
  return result;
}

Mat4 Translate(Mat4 m, const Vec3f &v) {
  // m[3] = m[0] * v + m[1] * v + m[2] * v + m[3];
  for (int i = 0; i < 4; ++i) {
    m.elements[3][i] = m.elements[0][i] * v.data[0] +
                       m.elements[1][i] * v.data[1] +
                       m.elements[2][i] * v.data[2] + m.elements[3][i];
  }
  return m;
}

Mat4 Rotate(const Mat4 &m, double angle, const Vec3f &v) {
  const double c = cos(angle);
  const double s = sin(angle);

  const Vec3f axis = Normalised(v);
  const Vec3f temp = axis * (1. - c);

  Mat4 rotate;
  rotate.elements[0][0] = c + temp.data[0] * axis.data[0];
  rotate.elements[0][1] = temp.data[0] * axis.data[1] + s * axis.data[2];
  rotate.elements[0][2] = temp.data[0] * axis.data[2] - s * axis.data[1];

  rotate.elements[1][0] = temp.data[1] * axis.data[0] - s * axis.data[2];
  rotate.elements[1][1] = c + temp.data[1] * axis.data[1];
  rotate.elements[1][2] = temp.data[1] * axis.data[2] + s * axis.data[0];

  rotate.elements[2][0] = temp.data[2] * axis.data[0] + s * axis.data[1];
  rotate.elements[2][1] = temp.data[2] * axis.data[1] - s * axis.data[0];
  rotate.elements[2][2] = c + temp.data[2] * axis.data[2];

  Mat4 result = m;
  for (int i = 0; i < 4; ++i) {
    result.elements[0][i] = m.elements[0][i] * rotate.elements[0][0] +
                            m.elements[1][i] * rotate.elements[0][1] +
                            m.elements[2][i] * rotate.elements[0][2];
    result.elements[1][i] = m.elements[0][i] * rotate.elements[1][0] +
                            m.elements[1][i] * rotate.elements[1][1] +
                            m.elements[2][i] * rotate.elements[1][2];
    result.elements[2][i] = m.elements[0][i] * rotate.elements[2][0] +
                            m.elements[1][i] * rotate.elements[2][1] +
                            m.elements[2][i] * rotate.elements[2][2];
  }
  return result;
}

Mat4 Ortho(double left, double right, double bottom, double top, double nearVal,
           double farVal) {
  const double rl = 1.0f / (right - left);
  const double tb = 1.0f / (top - bottom);
  const double fn = -1.0f / (farVal - nearVal);

  Mat4 dest = {0};
  dest.elements[0][0] = 2.0f * rl;
  dest.elements[1][1] = 2.0f * tb;
  dest.elements[2][2] = 2.0f * fn;
  dest.elements[3][0] = -(right + left) * rl;
  dest.elements[3][1] = -(top + bottom) * tb;
  dest.elements[3][2] = (farVal + nearVal) * fn;
  dest.elements[3][3] = 1.0f;

  return dest;
}

Mat4 Identity() {
  Mat4 m = {0};
  m.elements[0][0] = m.elements[1][1] = m.elements[2][2] = m.elements[3][3] =
      1.0f;
  return m;
}

Mat4 LookAt(const Vec3f &eye, const Vec3f &center, const Vec3f &up) {
  const Vec3f f = Normalised(center - eye);
  const Vec3f s = Normalised(CrossProduct(f, up));
  const Vec3f u = CrossProduct(s, f);

  Mat4 dest = {0};
  dest.elements[0][0] = s.data[0];
  dest.elements[0][1] = u.data[0];
  dest.elements[0][2] = -f.data[0];
  dest.elements[1][0] = s.data[1];
  dest.elements[1][1] = u.data[1];
  dest.elements[1][2] = -f.data[1];
  dest.elements[2][0] = s.data[2];
  dest.elements[2][1] = u.data[2];
  dest.elements[2][2] = -f.data[2];
  dest.elements[3][0] = -DotProduct(s, eye);
  dest.elements[3][1] = -DotProduct(u, eye);
  dest.elements[3][2] = DotProduct(f, eye);
  dest.elements[3][3] = 1.0f;
  return dest;
}

Mat4 Perspective(double fovy, double aspect, double zNear, double zFar) {
  const double f = 1.0f / tan(fovy * 0.5f);
  const double fn = 1.0f / (zNear - zFar);
  Mat4 dest = {0};
  dest.elements[0][0] = f / aspect;
  dest.elements[1][1] = f;
  dest.elements[2][2] = (zNear + zFar) * fn;
  dest.elements[2][3] = -1.0f;
  dest.elements[3][2] = 2.0f * zNear * zFar * fn;
  return dest;
}

Vec3f CalculateBBoxCenter(const BBox &box) {
  return (box.max + box.min) * 0.5f;
}

void Merge(BBox &box, const Vec3f &v) {
  box.min.x = (std::min)(box.min.x, v.x);
  box.min.y = (std::min)(box.min.y, v.y);
  box.min.z = (std::min)(box.min.z, v.z);
  box.max.x = (std::max)(box.max.x, v.x);
  box.max.y = (std::max)(box.max.y, v.y);
  box.max.z = (std::max)(box.max.z, v.z);
}

BBox Merge(const BBox &a, const BBox &b) {
  BBox result = a;
  Merge(result, b.min);
  Merge(result, b.max);
  return result;
}

bool IsBBoxValid(const BBox &a) {
  return a.max.x >= a.min.x && a.max.y >= a.min.y && a.max.z >= a.min.z;
}
Color GenerateColor() {
  static Color pregenerateList[]{
      Color{255, 0, 0, 255},   Color{255, 255, 0, 255}, Color{255, 0, 255, 255},
      Color{0, 255, 255, 255}, Color{0, 0, 255, 255},   Color{0, 255, 0, 255},
  };

  static const size_t listSize = sizeof(pregenerateList) / sizeof(Color);
  static size_t counter = 0;

  const Color result = pregenerateList[counter];
  counter = (counter + 1) % listSize;
  return result;
}

// Graphics
int32_t CompileShader(const char *shader, ShaderType type) {
  int32_t id = -1;
  switch (type) {
  case ShaderType::GEOMETRY: {
    id = glCreateShader(GL_GEOMETRY_SHADER);
  } break;
  case ShaderType::FRAGMENT: {
    id = glCreateShader(GL_FRAGMENT_SHADER);
  } break;
  case ShaderType::VERTEX: {
    id = glCreateShader(GL_VERTEX_SHADER);
  } break;
  }
  glShaderSource(id, 1, &shader, NULL);
  glCompileShader(id);
  GLint success;
  glGetShaderiv(id, GL_COMPILE_STATUS, &success);
  if (!success) {
    constexpr size_t LOG_ARRAY_SIZE = 256;
    char temp[LOG_ARRAY_SIZE] = {};
    glGetShaderInfoLog(id, sizeof(temp), NULL, temp);
    assert(false);
    return -1;
  }
  return id;
}

Program CreateProgram(const char *geometryShader, const char *vertexShader,
                      const char *fragmentShader) {
  Program program;
  int32_t gsId = -1, vsId = -1, fsId = -1;
  if (geometryShader) {
    gsId = CompileShader(geometryShader, ShaderType::GEOMETRY);
    if (gsId < 0) {
      return program;
    }
  }
  if (vertexShader) {
    vsId = CompileShader(vertexShader, ShaderType::VERTEX);
    if (vsId < 0) {
      return program;
    }
  }
  if (fragmentShader) {
    fsId = CompileShader(fragmentShader, ShaderType::FRAGMENT);
    if (fsId < 0) {
      return program;
    }
  }

  if (program.id != -1) {
    glDeleteProgram(program.id);
  }
  program.id = glCreateProgram();

  if (gsId >= 0)
    glAttachShader(program.id, gsId);
  if (fsId >= 0)
    glAttachShader(program.id, fsId);
  if (vsId >= 0)
    glAttachShader(program.id, vsId);
  glLinkProgram(program.id);

  GLint success = 0;
  glGetProgramiv(program.id, GL_LINK_STATUS, &success);
  program.valid = success;
  if (!success) {
    constexpr size_t LOG_ARRAY_SIZE = 256;
    char temp[LOG_ARRAY_SIZE] = {};
    glGetProgramInfoLog(program.id, sizeof(temp), NULL, temp);
    assert(false);
  }
  if (gsId >= 0)
    glDeleteShader(gsId);
  if (fsId >= 0)
    glDeleteShader(fsId);
  if (vsId >= 0)
    glDeleteShader(vsId);

  if (success) {
    if (geometryShader) {
      program.geometryShader = geometryShader;
    }
    if (vertexShader) {
      program.vertexShader = vertexShader;
    }
    if (fragmentShader) {
      program.fragmentShader = fragmentShader;
    }
  }
  return program;
}

uint32_t GenerateTexture() {
  uint32_t id;
  glGenTextures(1, &id);
  return id;
}

void UpdateTexture(uint32_t textureId, size_t width, size_t height,
                   Color *rgbaData) {
  glBindTexture(GL_TEXTURE_2D, textureId);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, rgbaData);
}

MeshRenderInfo CreateMeshRenderInfo(Mesh &mesh) {
  MeshRenderInfo result;

  const std::vector<Vec3f> vertexNormals =
      CalculateVertexNormals(mesh, BuildConnectivity(mesh));

  result.box = CalculateBBox(mesh);
  result.verticesCount = mesh.vertices.size();
  result.facesCount = mesh.faces.size();
  result.id = mesh.id;

  struct VertexInfo {
    Vec3f position;
    Vec3f normal;
  };
  std::vector<VertexInfo> vertices(result.verticesCount);
  for (size_t i = 0; i < result.verticesCount; i++) {
    vertices[i].position = mesh.vertices[i];
    vertices[i].normal = vertexNormals[i];
  }

  const uint32_t *indicies = (const uint32_t *)(mesh.faces.data());

  {
    uint32_t bufferId = 0;
    glGenVertexArrays(1, &bufferId);
    result.vertexBufferObject = bufferId;
  }
  glBindVertexArray(result.vertexBufferObject);
  {
    uint32_t bufferId = 0;
    glGenBuffers(1, &bufferId);
    result.vertexBufferId = bufferId;
  }
  {
    uint32_t bufferId = 0;
    glGenBuffers(1, &bufferId);
    result.elementBufferId = bufferId;
  }
  glBindBuffer(GL_ARRAY_BUFFER, result.vertexBufferId);
  glBufferData(GL_ARRAY_BUFFER, result.verticesCount * sizeof(VertexInfo),
               vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexInfo),
                        (void *)offsetof(VertexInfo, position));
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexInfo),
                        (void *)offsetof(VertexInfo, normal));
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, result.elementBufferId);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               result.facesCount * sizeof(Mesh::Triangle), indicies,
               GL_STATIC_DRAW);
  glBindVertexArray(0);
  return result;
}

void RenderMesh(const RenderBuffer &buffer, const Program &program,
                const MeshRenderInfo &info) {
  glBindFramebuffer(GL_FRAMEBUFFER, buffer.frameBufferId);
  glUseProgram(program.id);
  glBindVertexArray(info.vertexBufferObject);
  const size_t dataSize = 3 * info.facesCount;
  glDrawElements(GL_TRIANGLES, dataSize, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

RenderBuffer CreateRenderBuffer(size_t width, size_t height) {
  RenderBuffer result;
  result.width = width;
  result.height = height;
  // framebuffer configuration
  {
    uint32_t bufferId = 0;
    glGenFramebuffers(1, &bufferId);
    result.frameBufferId = bufferId;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, result.frameBufferId);
  // create a color attachment texture
  result.textureId = GenerateTexture();
  glBindTexture(GL_TEXTURE_2D, result.textureId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, result.width, result.height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         result.textureId, 0);
  // create a renderbuffer object for depth and stencil attachment (we won't
  // be sampling these)
  {
    uint32_t bufferId = 0;
    glGenRenderbuffers(1, &bufferId);
    result.renderBufferId = bufferId;
  }
  glBindRenderbuffer(GL_RENDERBUFFER, result.renderBufferId);
  // use a single renderbuffer object for both a depth AND stencil buffer.
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, result.width,
                        result.height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                            GL_RENDERBUFFER,
                            result.renderBufferId); // now actually attach it
  // now that we actually created the framebuffer and added all attachments we
  // want to check if it is actually complete now
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    assert(false);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  return result;
}

void ReizeRenderBuffer(RenderBuffer &buffer, size_t width, size_t heigth) {
  buffer.width = width;
  buffer.height = heigth;
  // framebuffer configuration
  glBindFramebuffer(GL_FRAMEBUFFER, buffer.frameBufferId);
  glBindTexture(GL_TEXTURE_2D, buffer.textureId);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, buffer.width, buffer.height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glBindRenderbuffer(GL_RENDERBUFFER, buffer.renderBufferId);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, buffer.width,
                        buffer.height);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ClearRenderBuffer(const RenderBuffer &buffer, Color c) {
  glBindFramebuffer(GL_FRAMEBUFFER, buffer.frameBufferId);
  glEnable(GL_DEPTH_TEST);
  glClearColor(c.r / 255.0f, c.g / 255.0f, c.b / 255.0f, c.a / 255.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glViewport(0, 0, buffer.width, buffer.height);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SetProgramUniformV3f(const Program &program, const char *name,
                          const float data[3]) {
  glUseProgram(program.id);
  glUniform3f(glGetUniformLocation(program.id, name), data[0], data[1],
              data[2]);
  glUseProgram(0);
}

void SetProgramUniformM4x4f(const Program &program, const char *name,
                            const float data[16]) {
  glUseProgram(program.id);
  glUniformMatrix4fv(glGetUniformLocation(program.id, name), 1, GL_FALSE, data);
  glUseProgram(0);
}

Mat4 Camera::GetViewMatrix() const { return viewMatrix; }

Mat4 Camera::GetProjectionMatrix(size_t width, size_t height) const {
  const float farClip = farClipRatio * lengthScale;
  const float nearClip = nearClipRatio * lengthScale;
  const float fovRad = Deg2Rad(fov);
  const float aspectRatio = width / (float)height;
  return Perspective(fovRad, aspectRatio, nearClip, farClip);
}

void Camera::FitBBox(const BBox &box) {
  center = CalculateBBoxCenter(box);
  lengthScale = Length(box.max - box.min);

  const Mat4 Tobj = ::Translate(Identity(), center * -1.0);
  const Mat4 Tcam =
      ::Translate(Identity(), Vec3f{0.0, 0.0f, -1.5f * lengthScale});

  viewMatrix = Tcam * Tobj;
  fov = DEFAULT_FOV;
  nearClipRatio = DEFAULT_NEAR_CLIP;
  farClipRatio = DEFAULT_FAR_CLIP;
}

void Camera::GetFrame(Vec3f &look, Vec3f &up, Vec3f &right) const {
  Mat3 r = {};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      r.elements[j][i] = viewMatrix.elements[i][j];
    }
  }
  look = r * Vec3f{0.0, 0.0, -1.0};
  up = r * Vec3f{0.0, 1.0, 0.0};
  right = r * Vec3f{1.0, 0.0, 0.0};
}

void Camera::Zoom(float amount) {
  if (amount == 0.0) {
    return;
  }
  // Translate the camera forwards and backwards
  const float movementScale = lengthScale * 0.1;
  static const Mat4 eye = Identity();
  const Mat4 camSpaceT =
      ::Translate(eye, Vec3f{0.0, 0.0, movementScale * amount});
  viewMatrix = camSpaceT * viewMatrix;
}

void Camera::Rotate(Vec2f start, Vec2f end) {
  if (Length(start - end) == 0) {
    return;
  }
  // Get frame
  Vec3f frameLookDir, frameUpDir, frameRightDir;
  GetFrame(frameLookDir, frameUpDir, frameRightDir);

  const Vec2f dragDelta = end - start;
  const double delTheta = 2.0 * dragDelta.x;
  const double delPhi = 2.0 * dragDelta.y;

  // Translate to center
  viewMatrix = ::Translate(viewMatrix, center);
  // Rotation about the vertical axis
  const Mat4 thetaCamR = ::Rotate(Identity(), delTheta, frameUpDir);
  viewMatrix = viewMatrix * thetaCamR;
  // Rotation about the horizontal axis
  const Mat4 phiCamR = ::Rotate(Identity(), -delPhi, frameRightDir);
  viewMatrix = viewMatrix * phiCamR;
  // Undo centering
  viewMatrix = ::Translate(viewMatrix, center * -1);
}

void Camera::Translate(Vec2f delta) {
  if (Length(delta) == 0) {
    return;
  }
  const double movementScale = lengthScale * 0.6;
  const Mat4 camSpaceT =
      ::Translate(Identity(), Vec3f{delta.x, delta.y, 0.0} * movementScale);
  viewMatrix = camSpaceT * viewMatrix;
}

//---------------------------Parsing-------------------------
struct Token {
  enum class Type {
    IDENTIFIER,
    NUMERIC_LITERAL,
    // operators
    OPERATOR_PLUS,
    OPERATOR_MINUS,
    OPERATOR_MULTIPLY,
    OPERATOR_DIVIDE,
    OPERATOR_POW,
    // functions
    FUNCTION_SIN,
    FUNCTION_COS,
    FUNCTION_TAN,
    FUNCTION_POW,
    //
    LEFT_PARAN,
    RIGHT_PARAN,
    COMMA,
  };

  Type type;
  // data.
  std::string name;      // when type == IDENTIFIER.
  double numericLiteral; // when type == NUMERIC_LITERAL.
};

// lexer

#define RETURN_IF_FALSE(expr)                                                  \
  do {                                                                         \
    if (!(expr)) {                                                             \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct Tokens {
  std::vector<Token> tokens;
  size_t cursor = 0;

  bool Next(Token &value) const {
    if (cursor < tokens.size()) {
      value = tokens[cursor];
      return true;
    }
    return false;
  }
  bool Consume() {
    if (cursor < tokens.size()) {
      cursor++;
      return true;
    }
    return false;
  }
  bool Expect(Token::Type type) {
    Token next;
    if (Next(next) && next.type == type) {
      Consume();
      return true;
    }
    return false;
  }
};

struct LexerResult {
  Tokens tokens;
  ExpressionParseErrorType error = ExpressionParseErrorType::PARSE_ERROR;
  size_t errorLocation = 0;
};

using CharBuffer = Buffer<const char>;
static bool CompareWordAndSkip(CharBuffer &buffer,
                               const std::string_view word) {
  const size_t length = word.size();
  if (buffer.cursor + length <= buffer.size) {
    for (size_t i = 0; i < length; ++i) {
      if (buffer.data[buffer.cursor + i] != word[i]) {
        return false;
      }
    }
    buffer.cursor += length;
    return true;
  }
  return false;
}
static bool ParseFloat(CharBuffer &buffer, double &result) {
  if (isdigit(buffer.Peek())) {
    std::string data;
    bool dotAdded = false;
    while (buffer.cursor < buffer.size &&
           (isdigit(buffer.Peek()) || buffer.Peek() == '.')) {
      if (buffer.Peek() == '.') {
        if (dotAdded) {
          return false;
        }
        dotAdded = true;
      }
      data.push_back(buffer.Peek());
      buffer.cursor++;
    }
    return std::from_chars(data.data(), data.data() + data.size(), result).ec ==
           std::errc();
  }
  return false;
}
static bool ParseIdentifier(CharBuffer &buffer, std::string &result) {
  if (isalpha(buffer.Peek())) {
    result.clear();
    while (buffer.cursor < buffer.size &&
           (isalnum(buffer.Peek()) || buffer.Peek() == '_')) {
      result.push_back(buffer.Peek());
      buffer.cursor++;
    }
    return true;
  }
  return false;
}
static LexerResult Tokenize(CharBuffer &buffer) {
  struct TokenString {
    Token::Type type;
    const char *string = nullptr;
  };

  static const TokenString tokensStrings[]{
      {Token::Type::FUNCTION_SIN, "sin"},
      {Token::Type::FUNCTION_COS, "cos"},
      {Token::Type::FUNCTION_TAN, "tan"},
      {Token::Type::FUNCTION_POW, "pow"},
      {Token::Type::OPERATOR_PLUS, "+"},
      {Token::Type::OPERATOR_MINUS, "-"},
      {Token::Type::OPERATOR_POW, "**"},
      {Token::Type::OPERATOR_MULTIPLY, "*"},
      {Token::Type::OPERATOR_DIVIDE, "/"},
      {Token::Type::LEFT_PARAN, "("},
      {Token::Type::RIGHT_PARAN, ")"},
      {Token::Type::COMMA, ","},
  };

  constexpr size_t count = std::size(tokensStrings);
  LexerResult result;
  while (buffer.cursor < buffer.size) {
    Token token;
    while (buffer.cursor < buffer.size && isblank(buffer.Peek())) {
      buffer.cursor++;
    }
    if (buffer.cursor >= buffer.size) {
      continue;
    }

    bool found = false;
    for (size_t i = 0; i < count; ++i) {
      if (CompareWordAndSkip(buffer, tokensStrings[i].string)) {
        token.type = tokensStrings[i].type;
        result.tokens.tokens.push_back(token);
        found = true;
        break;
      }
    }
    if (found) {
      continue;
    }
    if (ParseIdentifier(buffer, token.name)) {
      token.type = Token::Type::IDENTIFIER;
      result.tokens.tokens.push_back(token);
    } else if (ParseFloat(buffer, token.numericLiteral)) {
      token.type = Token::Type::NUMERIC_LITERAL;
      result.tokens.tokens.push_back(token);
    } else {
      result.error = ExpressionParseErrorType::PARSE_ERROR;
      result.errorLocation = buffer.cursor;
      return result;
    }
  }

  // specific to this application: only accept at most a single variable
  // functions and the variable name is x or y
  for (const Token token : result.tokens.tokens) {
    if (token.type == Token::Type::IDENTIFIER &&
        (token.name != "x" && token.name != "y" && token.name != "z")) {
      result.error = ExpressionParseErrorType::MULTIPLE_VARIABLES;
      return result;
    }
  }

  result.error = ExpressionParseErrorType::OK;
  return result;
}

// parser
static bool IsBinaryOperator(Token::Type type) {
  return type == Token::Type::OPERATOR_MULTIPLY ||
         type == Token::Type::OPERATOR_DIVIDE ||
         type == Token::Type::OPERATOR_PLUS ||
         type == Token::Type::OPERATOR_POW ||
         type == Token::Type::OPERATOR_MINUS;
}
static bool ParseP(Tokens &tokens, Expression &result);
static bool ParseE(Tokens &tokens, Expression &result);
static bool ParseE(Tokens &tokens, Expression &result) {
  Expression::Node node;
  node.children[0] = result.nodes.size();
  RETURN_IF_FALSE(ParseP(tokens, result));
  Token next;
  while (tokens.Next(next) && IsBinaryOperator(next.type)) {
    RETURN_IF_FALSE(tokens.Consume());
    if (next.type == Token::Type::OPERATOR_MULTIPLY) {
      node.type = Expression::NodeType::MULTIPLY;
    } else if (next.type == Token::Type::OPERATOR_PLUS) {
      node.type = Expression::NodeType::ADD;
    } else if (next.type == Token::Type::OPERATOR_DIVIDE) {
      node.type = Expression::NodeType::DIVIDE;
    } else if (next.type == Token::Type::OPERATOR_MINUS) {
      node.type = Expression::NodeType::SUB;
    } else if (next.type == Token::Type::OPERATOR_POW) {
      node.type = Expression::NodeType::POWER;
    }
    node.children[1] = result.nodes.size();
    RETURN_IF_FALSE(ParseP(tokens, result));
    result.rootNodeId = result.nodes.size();
    result.nodes.push_back(node);
  }
  return true;
}
static bool ParseP(Tokens &tokens, Expression &result) {
  Token next;
  if (tokens.Next(next) && next.type == Token::Type::IDENTIFIER) {
    RETURN_IF_FALSE(tokens.Consume());
    Expression::Node node;
    node.variable = next.name == "x" ? Expression::VariableType::X
                                     : Expression::VariableType::Y;
    node.type = Expression::NodeType::VARIABLE;
    result.nodes.push_back(node);
    return true;
  } else if (tokens.Next(next) && next.type == Token::Type::NUMERIC_LITERAL) {
    RETURN_IF_FALSE(tokens.Consume());
    Expression::Node node;
    node.constant = next.numericLiteral;
    node.type = Expression::NodeType::CONSTANAT;
    result.nodes.push_back(node);
    return true;
  } else if (tokens.Next(next) && next.type == Token::Type::FUNCTION_COS) {
    RETURN_IF_FALSE(tokens.Consume());
    RETURN_IF_FALSE(tokens.Expect(Token::Type::LEFT_PARAN));
    const size_t nodeId = result.nodes.size();
    result.nodes.push_back({});
    result.nodes[nodeId].type = Expression::NodeType::COS;
    result.nodes[nodeId].children[0] = result.nodes.size();
    RETURN_IF_FALSE(ParseE(tokens, result));
    RETURN_IF_FALSE(tokens.Expect(Token::Type::RIGHT_PARAN));
    return true;
  } else if (tokens.Next(next) && next.type == Token::Type::FUNCTION_SIN) {
    RETURN_IF_FALSE(tokens.Consume());
    RETURN_IF_FALSE(tokens.Expect(Token::Type::LEFT_PARAN));
    const size_t nodeId = result.nodes.size();
    result.nodes.push_back({});
    result.nodes[nodeId].type = Expression::NodeType::SIN;
    result.nodes[nodeId].children[0] = result.nodes.size();
    RETURN_IF_FALSE(ParseE(tokens, result));
    RETURN_IF_FALSE(tokens.Expect(Token::Type::RIGHT_PARAN));
    return true;
  } else if (tokens.Next(next) && next.type == Token::Type::FUNCTION_TAN) {
    RETURN_IF_FALSE(tokens.Consume());
    RETURN_IF_FALSE(tokens.Expect(Token::Type::LEFT_PARAN));
    const size_t nodeId = result.nodes.size();
    result.nodes.push_back({});
    result.nodes[nodeId].type = Expression::NodeType::TAN;
    result.nodes[nodeId].children[0] = result.nodes.size();
    RETURN_IF_FALSE(ParseE(tokens, result));
    RETURN_IF_FALSE(tokens.Expect(Token::Type::RIGHT_PARAN));
    return true;
  } else if (tokens.Next(next) && next.type == Token::Type::FUNCTION_POW) {
    RETURN_IF_FALSE(tokens.Consume());
    RETURN_IF_FALSE(tokens.Expect(Token::Type::LEFT_PARAN));
    const size_t nodeId = result.nodes.size();
    result.nodes.push_back({});
    result.nodes[nodeId].type = Expression::NodeType::POWER;
    result.nodes[nodeId].children[0] = result.nodes.size();
    RETURN_IF_FALSE(ParseE(tokens, result));
    RETURN_IF_FALSE(tokens.Expect(Token::Type::COMMA));
    result.nodes[nodeId].children[1] = result.nodes.size();
    RETURN_IF_FALSE(ParseE(tokens, result));
    RETURN_IF_FALSE(tokens.Expect(Token::Type::RIGHT_PARAN));
    return true;
  } else if (tokens.Next(next) && next.type == Token::Type::LEFT_PARAN) {
    RETURN_IF_FALSE(tokens.Consume());
    RETURN_IF_FALSE(ParseE(tokens, result));
    RETURN_IF_FALSE(tokens.Expect(Token::Type::RIGHT_PARAN));
    return true;
  }
  return false;
}
static bool ParseTokens(Tokens tokens, Expression &result) {
  if (ParseE(tokens, result)) {
    Token next;
    return !tokens.Next(next);
  }
  return false;
}

ExpressionParseErrorType GenerateExpression(const char *text,
                                            Expression &result) {
  CharBuffer buffer;
  buffer.data = text;
  buffer.size = strlen(text);
  const LexerResult lexerResults = Tokenize(buffer);
  switch (lexerResults.error) {
  case ExpressionParseErrorType::MULTIPLE_VARIABLES:
  case ExpressionParseErrorType::PARSE_ERROR:
    return lexerResults.error;
    break;
  }
  if (!ParseTokens(lexerResults.tokens, result)) {
    ExpressionParseErrorType::PARSE_ERROR;
  }
  return ExpressionParseErrorType::OK;
}

// Polygonisation
Mesh Polygonise(const FunctionType &expr, float isolevel, const float min[3],
                const float max[3], const float spacing[3]) {

  constexpr int16_t edgeTable[256] = {
      0x0,   0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905,
      0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00, 0x190, 0x99,  0x393, 0x29a,
      0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93,
      0xf99, 0xe90, 0x230, 0x339, 0x33,  0x13a, 0x636, 0x73f, 0x435, 0x53c,
      0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9,
      0x1a3, 0xaa,  0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6,
      0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569, 0x663, 0x76a, 0x66,  0x16f,
      0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
      0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff,  0x3f5, 0x2fc, 0xdfc, 0xcf5,
      0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a,
      0x256, 0x35f, 0x55,  0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53,
      0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
      0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9,
      0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc,  0x1c5, 0x2cf, 0x3c6,
      0x4ca, 0x5c3, 0x6c9, 0x7c0, 0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f,
      0xf55, 0xe5c, 0x15c, 0x55,  0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
      0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
      0xff,  0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a,
      0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x66,  0x76a, 0x663,
      0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
      0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa,  0x1a3, 0x2a9, 0x3a0, 0xd30, 0xc39,
      0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636,
      0x13a, 0x33,  0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f,
      0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99,  0x190,
      0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605,
      0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0};
  constexpr int8_t triTable[256][16] = {
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
      {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
      {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
      {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
      {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
      {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
      {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
      {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
      {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
      {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
      {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
      {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
      {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
      {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
      {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
      {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
      {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
      {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
      {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
      {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
      {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
      {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
      {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
      {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
      {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
      {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
      {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
      {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
      {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
      {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
      {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
      {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
      {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
      {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
      {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
      {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
      {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
      {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
      {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
      {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
      {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
      {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
      {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
      {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
      {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
      {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
      {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
      {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
      {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
      {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
      {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
      {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
      {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
      {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
      {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
      {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
      {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
      {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
      {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
      {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
      {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
      {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
      {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
      {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
      {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
      {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
      {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
      {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
      {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
      {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
      {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
      {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
      {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
      {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
      {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
      {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
      {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
      {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
      {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
      {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
      {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
      {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
      {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
      {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
      {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
      {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
      {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
      {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
      {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
      {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
      {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
      {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
      {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
      {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
      {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
      {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
      {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
      {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
      {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
      {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
      {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
      {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
      {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
      {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
      {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
      {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
      {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
      {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
      {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
      {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
      {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
      {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
      {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
      {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
      {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
      {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
      {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
      {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
      {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
      {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
      {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
      {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
      {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
      {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
      {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
      {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
      {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
      {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
      {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
      {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
      {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
      {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
      {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
      {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
      {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
      {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
      {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
      {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
      {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
      {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
      {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
      {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
      {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
      {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
      {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
      {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
      {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
      {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
      {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
      {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
      {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
      {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
      {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
      {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
      {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
      {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

  Mesh result;
  result.id = GenerateID();
  result.color = GenerateColor();
  result.visible = true;

  const size_t size[3] = {
      size_t((max[0] - min[0]) / spacing[0]) + 1,
      size_t((max[1] - min[1]) / spacing[1]) + 1,
      size_t((max[2] - min[2]) / spacing[2]) + 1,
  };

  std::vector<float> data(size[0] * size[1] * size[2]);
  Image3D image;
  image.data = data.data();
  for (size_t i = 0; i < 3; i++) {
    image.size[i] = size[i];
    image.origin[i] = min[i];
    image.spacing[i] = spacing[i];
  }

  for (size_t z = 0; z < size[2]; z++) {
    for (size_t y = 0; y < size[1]; y++) {
      for (size_t x = 0; x < size[0]; x++) {
        size_t index = z * (size[0] * size[1]) + y * size[0] + x;
        const double pt3d[3]{
            min[0] + spacing[0] * x,
            min[1] + spacing[1] * y,
            min[2] + spacing[2] * z,
        };
        image.At(x, y, z) = expr(pt3d[0], pt3d[1], pt3d[2]);
      }
    }
  }

  struct Grid {
    Vec3f p[8] = {};
    float val[8] = {};
  };

  std::unordered_map<Vec3f, size_t, Vec3fHash> pointsHashTable;

  auto InsertVertex = [&](const Vec3f &v) {
    auto itr = pointsHashTable.find(v);
    if (itr == pointsHashTable.end()) {
      size_t size = result.vertices.size();
      result.vertices.push_back(v);
      pointsHashTable[v] = size;
      return size;
    }
    return itr->second;
  };

  auto SetGridPoint = [&](Grid &grid, const Image3D &image, size_t i, size_t x,
                          size_t y, size_t z) {
    grid.val[i] = image.At(x, y, z);
    grid.p[i].x = image.origin[0] + x * image.spacing[0];
    grid.p[i].y = image.origin[1] + y * image.spacing[1];
    grid.p[i].z = image.origin[2] + z * image.spacing[2];
  };

  for (size_t z = 0; z < size[2] - 1; z++) {
    for (size_t y = 0; y < size[1] - 1; y++) {
      for (size_t x = 0; x < size[0] - 1; x++) {
        Grid grid;
        SetGridPoint(grid, image, 0, x, y, z);
        SetGridPoint(grid, image, 1, x + 1, y, z);
        SetGridPoint(grid, image, 2, x + 1, y, z + 1);
        SetGridPoint(grid, image, 3, x, y, z + 1);
        SetGridPoint(grid, image, 4, x, y + 1, z);
        SetGridPoint(grid, image, 5, x + 1, y + 1, z);
        SetGridPoint(grid, image, 6, x + 1, y + 1, z + 1);
        SetGridPoint(grid, image, 7, x, y + 1, z + 1);
        /*
            Determine the index into the edge table which
            tells us which vertices are inside of the surface
        */
        int cubeindex = 0;
        if (grid.val[0] < isolevel)
          cubeindex |= 1;
        if (grid.val[1] < isolevel)
          cubeindex |= 2;
        if (grid.val[2] < isolevel)
          cubeindex |= 4;
        if (grid.val[3] < isolevel)
          cubeindex |= 8;
        if (grid.val[4] < isolevel)
          cubeindex |= 16;
        if (grid.val[5] < isolevel)
          cubeindex |= 32;
        if (grid.val[6] < isolevel)
          cubeindex |= 64;
        if (grid.val[7] < isolevel)
          cubeindex |= 128;

        /* Cube is entirely in/out of the surface */
        if (edgeTable[cubeindex] == 0) {
          continue;
        }

        Vec3f vertexList[12] = {};

        /* Find the vertices where the surface intersects the cube */
        if (edgeTable[cubeindex] & 1)
          vertexList[0] =
              Lerp(isolevel, grid.p[0], grid.p[1], grid.val[0], grid.val[1]);
        if (edgeTable[cubeindex] & 2)
          vertexList[1] =
              Lerp(isolevel, grid.p[1], grid.p[2], grid.val[1], grid.val[2]);
        if (edgeTable[cubeindex] & 4)
          vertexList[2] =
              Lerp(isolevel, grid.p[2], grid.p[3], grid.val[2], grid.val[3]);
        if (edgeTable[cubeindex] & 8)
          vertexList[3] =
              Lerp(isolevel, grid.p[3], grid.p[0], grid.val[3], grid.val[0]);
        if (edgeTable[cubeindex] & 16)
          vertexList[4] =
              Lerp(isolevel, grid.p[4], grid.p[5], grid.val[4], grid.val[5]);
        if (edgeTable[cubeindex] & 32)
          vertexList[5] =
              Lerp(isolevel, grid.p[5], grid.p[6], grid.val[5], grid.val[6]);
        if (edgeTable[cubeindex] & 64)
          vertexList[6] =
              Lerp(isolevel, grid.p[6], grid.p[7], grid.val[6], grid.val[7]);
        if (edgeTable[cubeindex] & 128)
          vertexList[7] =
              Lerp(isolevel, grid.p[7], grid.p[4], grid.val[7], grid.val[4]);
        if (edgeTable[cubeindex] & 256)
          vertexList[8] =
              Lerp(isolevel, grid.p[0], grid.p[4], grid.val[0], grid.val[4]);
        if (edgeTable[cubeindex] & 512)
          vertexList[9] =
              Lerp(isolevel, grid.p[1], grid.p[5], grid.val[1], grid.val[5]);
        if (edgeTable[cubeindex] & 1024)
          vertexList[10] =
              Lerp(isolevel, grid.p[2], grid.p[6], grid.val[2], grid.val[6]);
        if (edgeTable[cubeindex] & 2048)
          vertexList[11] =
              Lerp(isolevel, grid.p[3], grid.p[7], grid.val[3], grid.val[7]);

        /* Create the triangle */
        for (int i = 0; triTable[cubeindex][i] != -1; i += 3) {
          Mesh::Triangle t;
          t.idx[0] = InsertVertex(vertexList[triTable[cubeindex][i]]);
          t.idx[1] = InsertVertex(vertexList[triTable[cubeindex][i + 1]]);
          t.idx[2] = InsertVertex(vertexList[triTable[cubeindex][i + 2]]);
          result.faces.push_back(t);
        }
      }
    }
  }
  return result;
}
#endif

#if defined(APPLICATION_CODE)

// surface with wireframes shaders
static const char *wires_fs = R"V0G0N(
#version 330 core
out vec4 FragColor;
in vec3 dist;
uniform vec3 objectColor;

const float lineWidth = 0.5;

float edgeFactor()
{
    vec3 d = fwidth(dist);
    vec3 f = step(d * lineWidth, dist);
    return min(min(f.x, f.y), f.z);
}

void main()
{
    gl_FragColor = vec4(min(vec3(edgeFactor()), objectColor), 1.0);
}
)V0G0N";

static const char *wires_vs = R"V0G0N(
#version 330 core
in vec4 position;
void main()
{
    gl_Position = position;
}
)V0G0N";

static const char *wires_gs = R"V0G0N(
#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
uniform mat4 view;
uniform mat4 projection;
out vec3 dist;

void main()
{
    mat4 mvp = projection * view;
    vec4 p0 = mvp * gl_in[0].gl_Position;
    vec4 p1 = mvp * gl_in[1].gl_Position;
    vec4 p2 = mvp * gl_in[2].gl_Position;

    dist = vec3(1, 0, 0);
    gl_Position = p0;
    EmitVertex();

    dist = vec3(0, 1, 0);
    gl_Position = p1;
    EmitVertex();

    dist = vec3(0, 0, 1);
    gl_Position = p2;
    EmitVertex();

    EndPrimitive();
}
)V0G0N";

static const size_t MAX_EXPRESSION_SIZE = 512;
static const ImVec4 RED(1, 0, 0, 1);
static const ImVec4 BLUE(41 / 255., 74 / 255., 122 / 255., 1);

struct Function {
  Expression expr;
  char name[MAX_EXPRESSION_SIZE] = {};
  Color color = GenerateColor();
  bool visible = true;
  float xRange[2] = {0, 1000};
  float yRange[2] = {0, 1000};
  float zRange[2] = {0, 1000};
  float sampleDistance[3] = {0.1, 0.1, 0.1};
};

struct View3DState {
  static constexpr size_t TEXTURE_WIDTH = 1024;
  static constexpr size_t TEXTURE_HEIGHT = 1024;
  size_t width = TEXTURE_WIDTH;
  size_t height = TEXTURE_HEIGHT;
  RenderBuffer buffer = CreateRenderBuffer(width, height);
  Color backgroundColor = Color{125, 125, 125, 255};
  Program program;
  bool redraw = true;
  Camera camera;
  std::vector<MeshRenderInfo> surfacesRenderInfo;

  void Fit() {
    BBox b;
    for (const MeshRenderInfo &info : surfacesRenderInfo) {
      b = Merge(b, info.box);
    }
    if (IsBBoxValid(b)) {
      camera.FitBBox(b);
      redraw = true;
    }
  }

  void Render(ImVec2 area, const std::vector<Mesh> &meshes) {
    ImGui::BeginChild("3D View", area);
    if (ImGui::IsWindowFocused()) {
      ImGuiIO &io = ImGui::GetIO();
      // If any mouse button is pressed, trigger a redraw
      if (ImGui::IsAnyMouseDown())
        redraw = true;

      // Handle scroll events for 3D view
      {
        const float offset = io.MouseWheel;
        if (offset) {
          camera.Zoom(offset);
          redraw = true;
        }
      }

      // Mouse inputs
      {
        // Process drags
        const bool dragLeft = ImGui::IsMouseDragging(0);
        // left takes priority, so only one can be true
        const bool dragRight = !dragLeft && ImGui::IsMouseDragging(1);

        if (dragLeft || dragRight) {
          const Vec2f dragDelta{io.MouseDelta.x / width,
                                io.MouseDelta.y / height};
          // exactly one of these will be true
          const bool isRotate = dragLeft && !io.KeyShift && !io.KeyCtrl;
          const bool isTranslate =
              (dragLeft && io.KeyShift && !io.KeyCtrl) || dragRight;
          const bool isDragZoom = dragLeft && io.KeyShift && io.KeyCtrl;

          if (isDragZoom) {
            camera.Zoom(dragDelta.y * 5);
          }
          if (isRotate) {
            const Vec2f currPos{2.f * (io.MousePos.x / width) - 1.0f,
                                2.f * ((height - io.MousePos.y) / height) -
                                    1.0f};
            camera.Rotate(currPos - (dragDelta * 2.0f), currPos);
          }
          if (isTranslate) {
            camera.Translate(dragDelta);
          }
        }
      }

      // reset best fit zoom.
      if (ImGui::IsKeyPressed(GLFW_KEY_R)) {
        Fit();
      }
    }

    const bool sizeChanged =
        width != size_t(area.x) || height != size_t(area.y);
    if (sizeChanged) {
      // update texture dimensions.
      width = area.x;
      height = area.y;
      ReizeRenderBuffer(buffer, width, height);
    }

    if (redraw || sizeChanged) {
      redraw = false;
      ClearRenderBuffer(buffer, backgroundColor);

      const Mat4 viewMatrix = camera.GetViewMatrix();
      const Mat4 projectionMatrix = camera.GetProjectionMatrix(width, height);
      float projectionMatrixData[16] = {};
      float viewMatrixData[16] = {};
      for (int i = 0; i < 16; ++i) {
        projectionMatrixData[i] = projectionMatrix.data[i];
        viewMatrixData[i] = viewMatrix.data[i];
      }
      // pass the parameters to the shader
      // lighting
      const Vec3f lightPos{1.2f, 1.0f, 2.0f};
      const Vec3f lighColour{1.2f, 1.0f, 2.0f};
      SetProgramUniformM4x4f(program, "projection", projectionMatrixData);
      SetProgramUniformM4x4f(program, "view", viewMatrixData);
      SetProgramUniformV3f(program, "lightPos", lightPos.data);
      SetProgramUniformV3f(program, "lightColor", lighColour.data);

      for (const MeshRenderInfo &info : surfacesRenderInfo) {
        for (const Mesh &mesh : meshes) {
          if (mesh.id == info.id && mesh.visible) {
            const float meshColor[3]{mesh.color.r / 255.f, mesh.color.g / 255.f,
                                     mesh.color.b / 255.f};
            SetProgramUniformV3f(program, "objectColor", meshColor);
            RenderMesh(buffer, program, info);
          }
        }
      }
    }
    ImGui::Image((ImTextureID)(intptr_t)buffer.textureId, area);
    ImGui::EndChild();
  }
};

struct State {
  struct GuiState {
    std::string errorLog;
    Function function;
  };

  std::vector<Mesh> meshes;
  View3DState view3d;
  GuiState gui;
  bool redraw = true;

  void Startup() {
    view3d.program = CreateProgram(wires_gs, wires_vs, wires_fs);
    assert(view3d.program.valid);
  }

  void Update() {
    ImGuiStyle &style = ImGui::GetStyle();
    style.FrameRounding = style.GrabRounding = 12;

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

    ImGui::Begin("Viewer", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_MenuBar |
                     ImGuiWindowFlags_NoBringToFrontOnFocus |
                     ImGuiWindowFlags_NoResize);
    const ImVec2 minPoint = ImGui::GetWindowContentRegionMin();
    const ImVec2 maxPoint = ImGui::GetWindowContentRegionMax();
    const float height = (maxPoint.y - minPoint.y);
    const float width = (maxPoint.x - minPoint.x);

    view3d.Render(ImVec2(width, height * 0.8), meshes);
    ImGui::BeginChild("Tools and Lists", ImVec2(-1, height * 0.2));
    {
      ImGui::Text("Function:");
      ImGui::SameLine();
      ImGui::InputText("##", gui.function.name, MAX_EXPRESSION_SIZE);

      ImGui::SameLine();
      if (ImGui::Button("Create torus")) {
        const float min[3]{-2, -2, -2};
        const float max[3]{2, 2, 2};
        const float spacing[3]{0.05, 0.05, 0.05};
        Mesh mesh = Polygonise(
            [](float x, float y, float z) {
              float x2 = x * x, y2 = y * y, z2 = z * z;
              float a = x2 + y2 + z2 + (0.5 * 0.5) - (0.1 * 0.1);
              return a * a - 4.0 * (0.5 * 0.5) * (y2 + z2);
            },
            0, min, max, spacing);
        mesh.name = gui.function.name;
        meshes.push_back(mesh);
        view3d.surfacesRenderInfo.push_back(CreateMeshRenderInfo(mesh));
        redraw = true;
        gui = {};
      }

      ImGui::SameLine();
      if (ImGui::Button("Create surface")) {
        if (ExpressionParseErrorType::OK ==
            GenerateExpression(gui.function.name, gui.function.expr)) {
          const float min[3]{gui.function.xRange[0], gui.function.yRange[0],
                             gui.function.zRange[0]};
          const float max[3]{gui.function.xRange[1], gui.function.yRange[1],
                             gui.function.zRange[1]};
          Mesh mesh = Polygonise(
              [&](float x, float y, float z) {
                return gui.function.expr.Evaluate(x, y, z);
              },
              0, min, max, gui.function.sampleDistance);
          mesh.name = gui.function.name;
          meshes.push_back(mesh);
          view3d.surfacesRenderInfo.push_back(CreateMeshRenderInfo(mesh));
          redraw = true;
          gui = {};
        } else {
          gui.errorLog = "Error parsing the expression";
        }
      }

      ImGui::Text("Range:");
      ImGui::BeginGroup();
      {
        ImGui::Text("X:");
        ImGui::SameLine();
        ImGui::InputFloat2("##0", gui.function.xRange);
        ImGui::SameLine();
        ImGui::Text("dX:");
        ImGui::SameLine();
        ImGui::InputFloat("##1", &gui.function.sampleDistance[0]);

        ImGui::Text("Y:");
        ImGui::SameLine();
        ImGui::InputFloat2("##2", gui.function.yRange);
        ImGui::SameLine();
        ImGui::Text("dY:");
        ImGui::SameLine();
        ImGui::InputFloat("##3", &gui.function.sampleDistance[1]);

        ImGui::Text("Z:");
        ImGui::SameLine();
        ImGui::InputFloat2("##4", gui.function.zRange);
        ImGui::SameLine();
        ImGui::Text("dZ:");
        ImGui::SameLine();
        ImGui::InputFloat("##5", &gui.function.sampleDistance[2]);
      }
      ImGui::EndGroup();

      if (gui.errorLog.size()) {
        ImGui::TextColored(RED, gui.errorLog.c_str());
      }

      ImGui::TextColored(BLUE, "Functions");
      for (Mesh &mesh : meshes) {
        ImGui::Spacing();
        if (ImGui::Checkbox(mesh.name.c_str(), &mesh.visible)) {
          redraw = true;
        }
        ImGui::SameLine();
        ImGui::Text("(%uz Faces, %uz Vertices)", mesh.faces.size(),
                    mesh.vertices.size());
        ImGui::SameLine();
        ImGui::PushID(mesh.name.c_str());
        float color[3] = {mesh.color.r / 255.0f, mesh.color.g / 255.0f,
                          mesh.color.b / 255.0f};
        if (ImGui::ColorEdit3("Colour", color,
                              ImGuiColorEditFlags_NoInputs |
                                  ImGuiColorEditFlags_NoLabel)) {
          mesh.color.r = color[0] * 255;
          mesh.color.g = color[1] * 255;
          mesh.color.b = color[2] * 255;
          redraw = true;
        }
        ImGui::PopID();
      }
    }
    ImGui::Text("Application average: %.1f FPS", ImGui::GetIO().Framerate);
    ImGui::EndChild();
    ImGui::End();
  }
};

int main(int argc, char **argv) {
  // Setup window
  if (!glfwInit()) {
    return EXIT_FAILURE;
  }
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(1280, 720, "Polygo", NULL, NULL);
  if (!window) {
    return EXIT_FAILURE;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // Initialize OpenGL loader
  if (gl3wInit() != 0) {
    return EXIT_FAILURE;
  }

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Load Fonts
  if (ImFont *font = io.Fonts->AddFontFromFileTTF("d:/Fira.ttf", 25.0f)) {
    io.FontDefault = font;
  }

  // Our state
  static const ImVec4 clearColor(0.45f, 0.55f, 0.60f, 1.00f);
  State state;
  state.Startup();

  // Main loop
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    state.Update();

    // Rendering
    ImGui::Render();
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();

  return EXIT_SUCCESS;
}
#endif
