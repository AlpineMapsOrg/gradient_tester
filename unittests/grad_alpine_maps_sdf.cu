/****************************************************************************
 *  Copyright (C) 2023 Adam Celarek (github.com/adam-ce, github.com/cg-tuwien)
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights to
 *  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 *  of the Software, and to permit persons to whom the Software is furnished to do so,
 *  subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 ****************************************************************************/

#include <stroke/unittest/gradcheck.h>

#include <catch2/catch_test_macros.hpp>
#include <stroke/gaussian.h>
#include <stroke/grad/gaussian.h>
#include <stroke/linalg.h>
#include <stroke/unittest/random_entity.h>

using Scalar = double;
using Vec2 = glm::vec<2, Scalar>;
using Vec3 = glm::vec<3, Scalar>;

namespace alpine {
struct GeomData {
    Vec2 p0;
    Vec2 p1;
    Vec2 p2;

    bool is_polygon;

    bool line_cap0;
    bool line_cap1;

    // normally in style, but for this test putting it here should be good enough
    Scalar line_width;
    Vec2 dash_info;
    bool round_line_caps;
};

Scalar sdf(const GeomData& data, const Vec2& uv, Scalar line_width, Vec2 dash_info, bool round_line_caps)
{
    Vec2 e0 = data.p1 - data.p0;
    Vec2 v0 = uv - data.p0;
    Vec2 v1 = uv - data.p1;

    Scalar h = glm::clamp(glm::dot(v0, e0) / glm::dot(e0, e0), Scalar(0), Scalar(1));
    Vec2 pq0 = v0 - e0 * h;

    Scalar poly_sign = 1.0;
    Scalar result = 1.0;
    Scalar mask = 1.0;

    if (data.is_polygon) {
        Vec2 e1 = data.p2 - data.p1;
        Vec2 e2 = data.p0 - data.p2;
        Vec2 v2 = uv - data.p2;
        Vec2 pq1 = v1 - e1 * glm::clamp(glm::dot(v1, e1) / glm::dot(e1, e1), Scalar(0), Scalar(1));
        Vec2 pq2 = v2 - e2 * glm::clamp(glm::dot(v2, e2) / glm::dot(e2, e2), Scalar(0), Scalar(1));
        Scalar s = glm::sign(e0.x * e2.y - e0.y * e2.x);
        Vec2 d0 = Vec2(glm::dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x));
        Vec2 d1 = Vec2(glm::dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
        Vec2 d2 = Vec2(glm::dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
        Vec2 d = min(min(d0, d1), d2);

        poly_sign = -glm::sign(d.y);
        result = d.x;
    } else {
        Scalar line_length = glm::length(e0);

        Scalar amount_dash_gap_pairs = glm::ceil(line_length/dash_info.y);
        // + 0.01 -> small delta to remove artifacts if there shouldn't be any dashes
        Scalar dash_period = cos(M_PI*h*amount_dash_gap_pairs*Scalar(2))+cos((Scalar(1)-dash_info.x)*M_PI)+Scalar(0.01);
        // tanh is used as a differentiable step function -> all values above 0 are mapped to +1, all below to -1
        // multiplication by big value ensures a quick transition at 0 +/- small delta
        Scalar dashes = tanh(dash_period*Scalar(500000.0));

        Scalar line_endings = 1.0;
        if(!round_line_caps)
        {
            if(data.line_cap0)
                line_endings *= dot(normalize(e0), v0);
            if(data.line_cap1)
                line_endings *= dot(normalize(-e0), v1);
        }
        line_endings = (tanh(line_endings*Scalar(500000.0))+Scalar(1)) / Scalar(2);

        mask = line_endings*dashes;

        result = glm::dot(pq0, pq0);
    }

    return (sqrt(result) * poly_sign) - (line_width * mask);
};

Vec3 sdf_with_grad(const GeomData& data, const Vec2& uv, Scalar incoming_grad)
{
    const Vec2 e0 = data.p1 - data.p0;
    const Vec2 v0 = uv - data.p0;
    const auto dot0 = glm::dot(v0, e0);
    const auto one_over_dot0 = 1 / glm::dot(e0, e0);
    const auto div0 = dot0 * one_over_dot0;
    const Vec2 pq0 = v0 - e0 * glm::clamp(div0, Scalar(0), Scalar(1));
    const auto dot_pq0_pq0 = glm::dot(pq0, pq0);

    Scalar poly_sign = 1.0;
    Scalar distance_sq = 1.0;

    Vec2 grad_uv = {};
    Vec2 grad_pq0 = {};
    if (data.is_polygon) {
        const Vec2 e1 = data.p2 - data.p1;
        const Vec2 e2 = data.p0 - data.p2;
        const Vec2 v1 = uv - data.p1;
        const Vec2 v2 = uv - data.p2;
        const auto dot1 = glm::dot(v1, e1);
        const auto dot2 = glm::dot(v2, e2);
        const auto one_over_dot1 = 1 / glm::dot(e1, e1);
        const auto one_over_dot2 = 1 / glm::dot(e2, e2);
        const auto div1 = dot1 * one_over_dot1;
        const auto div2 = dot2 * one_over_dot2;
        const auto clamp1 = glm::clamp(div1, Scalar(0), Scalar(1));
        const auto clamp2 = glm::clamp(div2, Scalar(0), Scalar(1));
        const Vec2 pq1 = v1 - e1 * clamp1;
        const Vec2 pq2 = v2 - e2 * clamp2;
        const Scalar s = glm::sign(e0.x * e2.y - e0.y * e2.x);
        const Vec2 d0 = Vec2(dot_pq0_pq0, s * (v0.x * e0.y - v0.y * e0.x));
        const Vec2 d1 = Vec2(glm::dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x));
        const Vec2 d2 = Vec2(glm::dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x));
        const Vec2 d = min(min(d0, d1), d2);

        poly_sign = -glm::sign(d.y);
        distance_sq = d.x;

        // gradient computation
        Scalar grad_d0_x = 0;
        Scalar grad_d1_x = 0;
        Scalar grad_d2_x = 0;

        if (d0.x <= d1.x && d0.x <= d2.x) {
            grad_d0_x = incoming_grad;
        } else if (d1.x < d0.x && d1.x <= d2.x) {
            grad_d1_x = incoming_grad;
        } else {
            assert(d2.x <= d0.x && d2.x <= d1.x);
            grad_d2_x = incoming_grad;
        }
        grad_pq0 = Scalar(2) * pq0 * grad_d0_x;
        const auto grad_pq1 = Scalar(2) * pq1 * grad_d1_x;
        const auto grad_pq2 = Scalar(2) * pq2 * grad_d2_x;

        auto grad_v1 = grad_pq1;
        auto grad_e1 = -grad_pq1 * clamp1;
        const auto grad_clamp1 = -glm::dot(e1, grad_pq1);

        auto grad_v2 = grad_pq2;
        auto grad_e2 = -grad_pq2 * clamp2;
        const auto grad_clamp2 = -glm::dot(e2, grad_pq2);

        const auto grad_div1 = stroke::grad::clamp(div1, Scalar(0), Scalar(1), grad_clamp1);

        const auto grad_div2 = stroke::grad::clamp(div2, Scalar(0), Scalar(1), grad_clamp2);

        const auto grad_dot1 = grad_div1 * one_over_dot1;

        const auto grad_dot2 = grad_div2 * one_over_dot2;

        grad_v1 += e1 * grad_dot1;
        grad_e1 += v1 * grad_dot1;

        grad_v2 += e2 * grad_dot2;
        grad_e2 += v2 * grad_dot2;

        grad_uv += grad_v1 + grad_v2;

    } else {
        grad_pq0 += Scalar(2) * pq0 * incoming_grad;
        distance_sq = dot_pq0_pq0;
    }

    Vec2 grad_v0 = grad_pq0;
    const auto grad_clamp = -glm::dot(grad_pq0, e0);
    const auto grad_div0 = stroke::grad::clamp(div0, Scalar(0), Scalar(1), grad_clamp);
    const auto grad_dot0 = grad_div0 * one_over_dot0;
    grad_v0 += e0 * grad_dot0;
    grad_uv += grad_v0;
    const auto sdf_val = sqrt(distance_sq) * poly_sign;

    return Vec3(sdf_val, grad_uv / (2 * sdf_val));
};

} // namespace alpine

TEST_CASE("alpine maps sdf")
{
    SECTION("intersect_with_ray_inv_C")
    {

        std::vector<alpine::GeomData> geomdata = {
            { { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 1.0 }, true, false, false, 0.0, {1.0, 1.0}, true }, // left bottom triangle ( one edge goes through center)
            { { 0.0, 0.0 }, { 0.5, 1.0 }, { 1.0, 0.5 }, true, false, false, 0.0, {1.0, 1.0}, true }, // triangle that encloses center
            { { 0.0, 0.0 }, { 0.2, 1.0 }, { 0.0, 1.0 }, true, false, false, 0.0, {1.0, 1.0}, true }, // triangle that is away from center

            { { 0.0, 0.0 }, { 0.0, 1.0 }, { 0.0, 0.0 }, false, false, false }, // horizontal line at uv border
            { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 }, false, false, false }, // vertical line at uv border
            { { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 0.0 }, false, false, false }, // diagonal line through center

            { { 0.0, 0.0 }, { 0.0, 1.0 }, { 0.0, 0.0 }, false, true, true}, // horizontal line at uv border
            { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 }, false, true, true }, // vertical line at uv border
            { { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 0.0 }, false, true, true }, // diagonal line through center

            { { 0.0, 0.0 }, { 0.0, 1.0 }, { 0.0, 0.0 }, false, true, false}, // horizontal line at uv border
            { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 0.0 }, false, false, true }, // vertical line at uv border
            { { 0.0, 0.0 }, { 1.0, 1.0 }, { 0.0, 0.0 }, false, true, false }, // diagonal line through center

            { { 0.06, -0.04 }, { 0.7, 0.62 }, { 0.46, 0.78 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.58, 1.1 }, { 0.9, 0.94 }, { -0.08, 0.38 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.22, 0.54 }, { 0.7, 0 }, { 0.62, 0.82 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.82, 0.8 }, { 0.52, -0.02 }, { 0.32, 1.12 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.44, 0.22 }, { 0.82, 0.04 }, { 0.78, 0.38 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.86, 0.88 }, { -0.02, 0.12 }, { 0.4, 0.8 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.78, 1.16 }, { -0.06, 0.1 }, { 0.5, 0.38 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 1, 0.24 }, { -0.06, 0.84 }, { 0.32, -0.04 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.54, 1.02 }, { 0.94, 1.02 }, { 0.86, 1.12 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.8, 0.6 }, { 0.28, 0.82 }, { 1, 0.98 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0, 0.56 }, { 0.18, 0.54 }, { 0.92, -0.02 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.76, 1.02 }, { 0.14, 0.88 }, { 0.22, 1.18 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 1.16, 0.66 }, { 0.42, 1.1 }, { 0.64, 0.08 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.44, 0.44 }, { 0.68, 0.56 }, { 1.16, 0.22 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { -0.1, 0.26 }, { 1, 0.18 }, { 0.5, 1.04 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.24, 1.16 }, { 0.7, 0.56 }, { 0.14, 0.82 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0, 0.28 }, { 1.18, 0.58 }, { 0.96, 1.06 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.7, 0.78 }, { 0.54, 0.48 }, { 0.66, 0.82 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0.22, -0.1 }, { 0.78, 1 }, { 0.86, 0.82 }, true, false, false, 0.0, {1.0, 1.0}, true },
            { { 0, 0.56 }, { 0.18, 0.54 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, true },
            { { 0.76, 1.02 }, { 0.14, 0.88 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, true },
            { { 1.16, 0.66 }, { 0.42, 1.1 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, true },
            { { 0.44, 0.44 }, { 0.68, 0.56 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, true },
            { { -0.1, 0.26 }, { 1, 0.18 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, true },
            // line endings -> but not at line segment end
            { { 0.24, 1.16 }, { 0.7, 0.56 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, false },
            { { 0, 0.28 }, { 1.18, 0.58 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, false },
            { { 0.7, 0.78 }, { 0.54, 0.48 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, false },
            { { 0.22, -0.1 }, { 0.78, 1 }, { 0, 0 }, false, false, false, 0.05, {1.0, 1.0}, false },

            // line endings
            { { 0, 0.56 }, { 0.18, 0.54 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 0.76, 1.02 }, { 0.14, 0.88 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 1.16, 0.66 }, { 0.42, 1.1 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 0.44, 0.44 }, { 0.68, 0.56 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { -0.1, 0.26 }, { 1, 0.18 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 0.24, 1.16 }, { 0.7, 0.56 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 0, 0.28 }, { 1.18, 0.58 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 0.7, 0.78 }, { 0.54, 0.48 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },
            { { 0.22, -0.1 }, { 0.78, 1 }, { 0, 0 }, false, true, true, 0.05, {1.0, 1.0}, false },

            // dashes
            { { 0, 0.56 }, { 0.18, 0.54 }, { 0, 0 }, false, false, false, 0.05, {0.5, 0.05}, true },
            { { 0.76, 1.02 }, { 0.14, 0.88 }, { 0, 0 }, false, false, false, 0.05, {0.7, 0.05}, true },
            { { 1.16, 0.66 }, { 0.42, 1.1 }, { 0, 0 }, false, false, false, 0.05, {0.5, 0.5}, true },
            { { 0.44, 0.44 }, { 0.68, 0.56 }, { 0, 0 }, false, false, false, 0.05, {0.5, 0.002}, true },
            { { -0.1, 0.26 }, { 1, 0.18 }, { 0, 0 }, false, false, false, 0.05, {0.5, 0.05}, true },
            { { 0.24, 1.16 }, { 0.7, 0.56 }, { 0, 0 }, false, false, false, 0.05, {0.5, 0.0005}, true },
            { { 0, 0.28 }, { 1.18, 0.58 }, { 0, 0 }, false, false, false, 0.05, {0.2, 0.05}, true },
            { { 0.7, 0.78 }, { 0.54, 0.48 }, { 0, 0 }, false, false, false, 0.05, {0.3, 0.05}, true },
            { { 0.22, -0.1 }, { 0.78, 1 }, { 0, 0 }, false, false, false, 0.05, {0.4, 0.005}, true },

            // dashes + line endings
            { { 0, 0.56 }, { 0.18, 0.54 }, { 0, 0 }, false, true, true, 0.05, {0.5, 0.05}, false },
            { { 0.76, 1.02 }, { 0.14, 0.88 }, { 0, 0 }, false, true, true, 0.05, {0.7, 0.05}, false },
            { { 1.16, 0.66 }, { 0.42, 1.1 }, { 0, 0 }, false, true, true, 0.05, {0.5, 0.5}, false },
            { { 0.44, 0.44 }, { 0.68, 0.56 }, { 0, 0 }, false, true, true, 0.05, {0.5, 0.002}, false },
            { { -0.1, 0.26 }, { 1, 0.18 }, { 0, 0 }, false, true, true, 0.05, {0.5, 0.05}, false },
            { { 0.24, 1.16 }, { 0.7, 0.56 }, { 0, 0 }, false, true, true, 0.05, {0.5, 0.0005}, false },
            { { 0, 0.28 }, { 1.18, 0.58 }, { 0, 0 }, false, true, true, 0.05, {0.2, 0.05}, false },
            { { 0.7, 0.78 }, { 0.54, 0.48 }, { 0, 0 }, false, true, true, 0.05, {0.3, 0.05}, false },
            { { 0.22, -0.1 }, { 0.78, 1 }, { 0, 0 }, false, true, true, 0.05, {0.4, 0.005}, false },
        };

        whack::random::HostGenerator<Scalar> rnd;

        for (const auto& data : geomdata) {
            for (int i = 0; i < 10; ++i) {
                const auto fun = [&](const whack::Tensor<Scalar, 1>& input) {
                    const auto uv = stroke::extract<Vec2>(input);
                    const auto d = alpine::sdf(data, uv, data.line_width, data.dash_info, data.round_line_caps);
                    const auto d2 = sdf_with_grad(data, uv, 1);
                    CHECK(Catch::Approx(d) == d2.x);
                    return stroke::pack_tensor<Scalar>(d);
                };

                const auto fun_grad = [&](const whack::Tensor<Scalar, 1>& input, const whack::Tensor<Scalar, 1>& grad_output) {
                    const auto uv = stroke::extract<Vec2>(input);
                    const auto grad_incoming = stroke::extract<Scalar>(grad_output);

                    const auto grad_outgoing = alpine::sdf_with_grad(data, uv, grad_incoming);

                    return stroke::pack_tensor<Scalar>(grad_outgoing.y, grad_outgoing.z);
                };

                const auto uv = rnd.uniform2();
                const auto test_data = stroke::pack_tensor<Scalar>(uv);
                stroke::check_gradient(fun, fun_grad, test_data, Scalar(0.000001));
            }
        }
    }
}
