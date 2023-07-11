var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf, __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: !0 });
}, __copyProps = (to, from, except, desc) => {
  if (from && typeof from == "object" || typeof from == "function")
    for (let key of __getOwnPropNames(from))
      !__hasOwnProp.call(to, key) && key !== except && __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: !0 }) : target,
  mod
)), __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: !0 }), mod);

// <stdin>
var stdin_exports = {};
__export(stdin_exports, {
  assets: () => assets_manifest_default,
  assetsBuildDirectory: () => assetsBuildDirectory,
  dev: () => dev,
  entry: () => entry,
  future: () => future,
  publicPath: () => publicPath,
  routes: () => routes
});
module.exports = __toCommonJS(stdin_exports);

// app/entry.server.jsx
var entry_server_exports = {};
__export(entry_server_exports, {
  default: () => handleRequest
});
var import_node_stream = require("node:stream"), import_node = require("@remix-run/node"), import_react = require("@remix-run/react"), import_isbot = __toESM(require("isbot")), import_server = require("react-dom/server"), import_jsx_dev_runtime = require("react/jsx-dev-runtime"), ABORT_DELAY = 5e3;
function handleRequest(request, responseStatusCode, responseHeaders, remixContext, loadContext) {
  return (0, import_isbot.default)(request.headers.get("user-agent")) ? handleBotRequest(
    request,
    responseStatusCode,
    responseHeaders,
    remixContext
  ) : handleBrowserRequest(
    request,
    responseStatusCode,
    responseHeaders,
    remixContext
  );
}
function handleBotRequest(request, responseStatusCode, responseHeaders, remixContext) {
  return new Promise((resolve, reject) => {
    let shellRendered = !1, { pipe, abort } = (0, import_server.renderToPipeableStream)(
      /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(
        import_react.RemixServer,
        {
          context: remixContext,
          url: request.url,
          abortDelay: ABORT_DELAY
        },
        void 0,
        !1,
        {
          fileName: "app/entry.server.jsx",
          lineNumber: 47,
          columnNumber: 7
        },
        this
      ),
      {
        onAllReady() {
          shellRendered = !0;
          let body = new import_node_stream.PassThrough();
          responseHeaders.set("Content-Type", "text/html"), resolve(
            new import_node.Response(body, {
              headers: responseHeaders,
              status: responseStatusCode
            })
          ), pipe(body);
        },
        onShellError(error) {
          reject(error);
        },
        onError(error) {
          responseStatusCode = 500, shellRendered && console.error(error);
        }
      }
    );
    setTimeout(abort, ABORT_DELAY);
  });
}
function handleBrowserRequest(request, responseStatusCode, responseHeaders, remixContext) {
  return new Promise((resolve, reject) => {
    let shellRendered = !1, { pipe, abort } = (0, import_server.renderToPipeableStream)(
      /* @__PURE__ */ (0, import_jsx_dev_runtime.jsxDEV)(
        import_react.RemixServer,
        {
          context: remixContext,
          url: request.url,
          abortDelay: ABORT_DELAY
        },
        void 0,
        !1,
        {
          fileName: "app/entry.server.jsx",
          lineNumber: 97,
          columnNumber: 7
        },
        this
      ),
      {
        onShellReady() {
          shellRendered = !0;
          let body = new import_node_stream.PassThrough();
          responseHeaders.set("Content-Type", "text/html"), resolve(
            new import_node.Response(body, {
              headers: responseHeaders,
              status: responseStatusCode
            })
          ), pipe(body);
        },
        onShellError(error) {
          reject(error);
        },
        onError(error) {
          responseStatusCode = 500, shellRendered && console.error(error);
        }
      }
    );
    setTimeout(abort, ABORT_DELAY);
  });
}

// app/root.jsx
var root_exports = {};
__export(root_exports, {
  default: () => App,
  links: () => links
});
var import_react2 = require("@remix-run/react");

// app/styles/main.css
var main_default = "/build/_assets/main-XQIXJFXJ.css";

// app/tailwind/tailwind.css
var tailwind_default = "/build/_assets/tailwind-KVIRLIZX.css";

// app/root.jsx
var import_jsx_dev_runtime2 = require("react/jsx-dev-runtime");
function App() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)("html", { lang: "en", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)("head", { children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)("meta", { charSet: "utf-8" }, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 18,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)("meta", { name: "viewport", content: "width=device-width,initial-scale=1" }, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 19,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)(import_react2.Meta, {}, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 20,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)(import_react2.Links, {}, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 21,
        columnNumber: 9
      }, this)
    ] }, void 0, !0, {
      fileName: "app/root.jsx",
      lineNumber: 17,
      columnNumber: 7
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)("body", { children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)(import_react2.Outlet, {}, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 27,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)(import_react2.ScrollRestoration, {}, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 28,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)(import_react2.Scripts, {}, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 29,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime2.jsxDEV)(import_react2.LiveReload, {}, void 0, !1, {
        fileName: "app/root.jsx",
        lineNumber: 30,
        columnNumber: 9
      }, this)
    ] }, void 0, !0, {
      fileName: "app/root.jsx",
      lineNumber: 23,
      columnNumber: 7
    }, this)
  ] }, void 0, !0, {
    fileName: "app/root.jsx",
    lineNumber: 16,
    columnNumber: 5
  }, this);
}
function links() {
  return [{ rel: "stylesheet", href: main_default }, { rel: "stylesheet", href: tailwind_default }];
}

// app/routes/_index.jsx
var index_exports = {};
__export(index_exports, {
  default: () => Index
});
var import_react16 = require("@remix-run/react");

// app/Components/slider.jsx
var import_react3 = require("react"), import_react4 = require("@material-tailwind/react");
var import_jsx_dev_runtime3 = require("react/jsx-dev-runtime");
function Slider() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Carousel, { className: "h-[100vh]", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "relative h-full w-full", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(
        "img",
        {
          src: "./images/1.jpg",
          alt: "image 1",
          className: "h-full w-full object-cover"
        },
        void 0,
        !1,
        {
          fileName: "app/Components/slider.jsx",
          lineNumber: 11,
          columnNumber: 13
        },
        this
      ),
      /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "absolute inset-0 grid h-full w-full place-items-center bg-black/75", children: /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "w-3/4 text-center md:w-2/4", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Typography, { variant: "h1", color: "white", className: "mb-4 md:text-4xl lg:text-5xl", children: "Medical Record Management" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 17,
          columnNumber: 21
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Typography, { variant: "lead", color: "white", className: "mb-12 opacity-80", children: "Lorem ipsum dolor sit, amet consectetur adipisicing elit. Eos libero fuga obcaecati natus facere! Fugit amet in maxime iure dignissimos obcaecati sit assumenda eius dolor? Cumque fugiat quod explicabo voluptates!" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 20,
          columnNumber: 21
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "flex justify-center gap-2", children: /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Button, { size: "lg", color: "white", children: "Get Started" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 26,
          columnNumber: 25
        }, this) }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 25,
          columnNumber: 21
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 16,
        columnNumber: 17
      }, this) }, void 0, !1, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 15,
        columnNumber: 13
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/slider.jsx",
      lineNumber: 10,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "relative h-full w-full", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(
        "img",
        {
          src: "./images/2.jpg",
          alt: "",
          className: "h-full w-full object-cover"
        },
        void 0,
        !1,
        {
          fileName: "app/Components/slider.jsx",
          lineNumber: 34,
          columnNumber: 13
        },
        this
      ),
      /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "absolute inset-0 grid h-full w-full place-items-center bg-black/75", children: /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "w-3/4 text-center md:w-2/4", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Typography, { variant: "h1", color: "white", className: "mb-4 md:text-4xl lg:text-5xl", children: "The Beauty of Nature" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 38,
          columnNumber: 21
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Typography, { variant: "lead", color: "white", className: "mb-12 opacity-80", children: "Lorem ipsum dolor sit, amet consectetur adipisicing elit. Eos libero fuga obcaecati natus facere! Fugit amet in maxime iure dignissimos obcaecati sit assumenda eius dolor? Cumque fugiat quod explicabo voluptates!" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 41,
          columnNumber: 21
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "flex justify-center gap-2", children: /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Button, { size: "lg", color: "white", children: "Get Started" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 47,
          columnNumber: 25
        }, this) }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 46,
          columnNumber: 21
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 37,
        columnNumber: 17
      }, this) }, void 0, !1, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 36,
        columnNumber: 13
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/slider.jsx",
      lineNumber: 33,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "relative h-full w-full", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("img", { src: "./images/3.jpg", alt: "", className: "h-full w-full object-cover" }, void 0, !1, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 55,
        columnNumber: 13
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "absolute inset-0 grid h-full w-full place-items-center bg-black/75", children: /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "w-3/4 text-center md:w-2/4", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Typography, { variant: "h1", color: "white", className: "mb-4 md:text-4xl lg:text-5xl", children: "The Beauty of Nature" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 58,
          columnNumber: 21
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Typography, { variant: "lead", color: "white", className: "mb-12 opacity-80", children: "Lorem ipsum dolor sit, amet consectetur adipisicing elit. Eos libero fuga obcaecati natus facere! Fugit amet in maxime iure dignissimos obcaecati sit assumenda eius dolor? Cumque fugiat quod explicabo voluptates!" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 61,
          columnNumber: 21
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)("div", { className: "flex justify-center gap-2", children: /* @__PURE__ */ (0, import_jsx_dev_runtime3.jsxDEV)(import_react4.Button, { size: "lg", color: "white", children: "Get Started" }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 67,
          columnNumber: 25
        }, this) }, void 0, !1, {
          fileName: "app/Components/slider.jsx",
          lineNumber: 66,
          columnNumber: 21
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 57,
        columnNumber: 17
      }, this) }, void 0, !1, {
        fileName: "app/Components/slider.jsx",
        lineNumber: 56,
        columnNumber: 13
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/slider.jsx",
      lineNumber: 54,
      columnNumber: 9
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/slider.jsx",
    lineNumber: 9,
    columnNumber: 5
  }, this);
}

// app/Components/pricing.jsx
var import_react5 = require("react"), import_react6 = require("@material-tailwind/react"), import_jsx_dev_runtime4 = require("react/jsx-dev-runtime");
function Pricing(props) {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("div", { id: "pricing", className: "relative grid gap-16 h-full p-8 grid-cols-1 grid-rows-auto", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("div", { className: "text-white min-h-[100px] flex flex-col gap-6 w-full justify-center place-items-center text-center", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)(import_react6.Typography, { variant: "h1", color: "White", className: "text-bold text-3xl", children: "Pricing plans for teams of all sizes" }, void 0, !1, {
        fileName: "app/Components/pricing.jsx",
        lineNumber: 8,
        columnNumber: 13
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-[18px] w-full md:w-2/4 ", children: "Choose an affordable plan that\u2019s packed with the best features for engaging your audience, creating customer loyalty, and driving sales." }, void 0, !1, {
        fileName: "app/Components/pricing.jsx",
        lineNumber: 11,
        columnNumber: 13
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/pricing.jsx",
      lineNumber: 7,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("div", { className: "grid gap-6 text-white m-auto lg:grid-cols-3", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("div", { className: "flex flex-col border-[1px] border-gray-700 rounded-[30px] min-h-[70vh] md:min-h-[50vh] w-[50vh]  lg:min-h-[90vh] p-8 hover:w-70 cursor-pointe", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-blue-700 text-[18px] lg:text-[18px] md:text-3xl", children: "Basic Plan" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 19,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: "mt-4 md:text-[18px]", children: "The essentials to provide your best work for clients" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 22,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: " flex mt-4 mb-6 place-items-baseline", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-5xl mr-2", children: "$ 10 /" }, void 0, !1, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 24,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { children: "month" }, void 0, !1, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 24,
            columnNumber: 76
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 23,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: "text-bold text-blue-700 mt-6 md:text-[20px] lg:text-sm ", children: "What's Included" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 26,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("ul", { role: "list", className: "mt-6 mb-8 grid grid-cols-1 gap-4 text-sm leading-6 md:text-[20px] lg:text-sm", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 30,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 29,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 28,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 36,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 35,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 34,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 42,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 41,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 40,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 48,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 47,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 46,
            columnNumber: 21
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 27,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("button", { className: "rounded w-full h-10 m-auto lg:h-10 bg-blue-700 md:h-20", children: "Start now" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 53,
          columnNumber: 17
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/pricing.jsx",
        lineNumber: 18,
        columnNumber: 13
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("div", { className: "flex flex-col border-[1px] border-gray-700 rounded-[30px] min-h-[70vh] md:min-h-[50vh] w-[50vh]  lg:min-h-[90vh] p-8 hover:w-70 cursor-pointe", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-blue-700 text-[18px] lg:text-[18px] md:text-3xl", children: "Compact Plan" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 56,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: "mt-4 md:text-[18px]", children: "The essentials to provide your best work for clients" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 59,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: " flex mt-4 mb-6 place-items-baseline", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-5xl mr-2", children: "$ 20 /" }, void 0, !1, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 61,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { children: "month" }, void 0, !1, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 61,
            columnNumber: 76
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 60,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: "text-bold text-blue-700 mt-6 md:text-[20px] lg:text-sm ", children: "What's Included" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 63,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("ul", { role: "list", className: "mt-6 mb-8 grid grid-cols-1 gap-4 text-sm leading-6 md:text-[20px] lg:text-sm", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 67,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 66,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 65,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 73,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 72,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 71,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 79,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 78,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 77,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 85,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 84,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 83,
            columnNumber: 21
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 64,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("button", { className: "rounded w-full h-10 m-auto lg:h-10 bg-blue-700 md:h-20", children: "Start now" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 90,
          columnNumber: 17
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/pricing.jsx",
        lineNumber: 55,
        columnNumber: 13
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("div", { className: "flex flex-col border-[1px] border-gray-700 rounded-[30px] min-h-[70vh] md:min-h-[50vh] w-[50vh]  lg:min-h-[90vh] p-8 hover:w-70 cursor-pointe", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-blue-700 text-[18px] lg:text-[18px] md:text-3xl", children: "Premium Plan" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 93,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: "mt-4 md:text-[18px]", children: "The essentials to provide your best work for clients" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 96,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: " flex mt-4 mb-6 place-items-baseline", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { className: "text-bold text-5xl mr-2", children: "$ 30 /" }, void 0, !1, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 98,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("span", { children: "month" }, void 0, !1, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 98,
            columnNumber: 76
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 97,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("p", { className: "text-bold text-blue-700 mt-6 md:text-[20px] lg:text-sm ", children: "What's Included" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 100,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("ul", { role: "list", className: "mt-6 mb-8 grid grid-cols-1 gap-4 text-sm leading-6 md:text-[20px] lg:text-sm", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 104,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 103,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 102,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 110,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 109,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 108,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 116,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 115,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 114,
            columnNumber: 21
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("li", { className: "flex gap-x-3", children: [
            /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("svg", { class: "h-6 w-5 flex-none text-indigo-600", viewBox: "0 0 20 20", fill: "currentColor", "aria-hidden": "true", children: /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("path", { "fill-rule": "evenodd", d: "M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.05-.143z", "clip-rule": "evenodd" }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 122,
              columnNumber: 25
            }, this) }, void 0, !1, {
              fileName: "app/Components/pricing.jsx",
              lineNumber: 121,
              columnNumber: 25
            }, this),
            "Member resources"
          ] }, void 0, !0, {
            fileName: "app/Components/pricing.jsx",
            lineNumber: 120,
            columnNumber: 21
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 101,
          columnNumber: 17
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime4.jsxDEV)("button", { className: "rounded w-full h-10 m-auto lg:h-10 bg-blue-700 md:h-20", children: "Start now" }, void 0, !1, {
          fileName: "app/Components/pricing.jsx",
          lineNumber: 127,
          columnNumber: 17
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/pricing.jsx",
        lineNumber: 92,
        columnNumber: 13
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/pricing.jsx",
      lineNumber: 17,
      columnNumber: 9
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/pricing.jsx",
    lineNumber: 6,
    columnNumber: 5
  }, this);
}
var pricing_default = Pricing;

// app/Components/features.jsx
var import_react7 = require("react"), import_react8 = require("@material-tailwind/react"), import_solid = require("@heroicons/react/24/solid"), import_jsx_dev_runtime5 = require("react/jsx-dev-runtime"), features = [
  {
    name: "Patient Record Management",
    decription: "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
    icon: import_solid.BookOpenIcon
  },
  {
    name: "Billing",
    decription: "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
    icon: import_solid.CalculatorIcon
  },
  {
    name: "Reports and Analysis",
    decription: "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
    icon: import_solid.ChartPieIcon
  },
  {
    name: "Stock Management",
    decription: "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at.In mi viverra elit nunc.",
    icon: import_solid.Square3Stack3DIcon
  }
];
function Features() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("div", { id: "features", className: "relative bg-white grid gap-16 min-h-[100vh] p-8 grid-cols-1 grid-rows-auto", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("div", { className: "text-black min-h-[100px] flex flex-col gap-6 w-full justify-center place-items-center text-center", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)(import_react8.Typography, { variant: "h1", color: "White", className: "text-bold text-3xl", children: "Everything you need to get Started" }, void 0, !1, {
        fileName: "app/Components/features.jsx",
        lineNumber: 44,
        columnNumber: 13
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("span", { className: "text-bold text-[18px] w-full md:w-2/4 ", children: "Quis tellus eget adipiscing convallis sit sit eget aliquet quis. Suspendisse eget egestas a elementum pulvinar et feugiat blandit at. In mi viverra elit nunc." }, void 0, !1, {
        fileName: "app/Components/features.jsx",
        lineNumber: 47,
        columnNumber: 13
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/features.jsx",
      lineNumber: 43,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("div", { className: "grid mx-auto px-10 gap-6 lg:grid-cols-2", children: [
      features.map((feature) => /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("div", { className: "flex m-4", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("h1", { className: "block p-4  text-white", children: /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)(feature.icon, { className: "w-7 h-7 p-1 bg-[#062D99] rounded " }, void 0, !1, {
          fileName: "app/Components/features.jsx",
          lineNumber: 59,
          columnNumber: 29
        }, this) }, void 0, !1, {
          fileName: "app/Components/features.jsx",
          lineNumber: 58,
          columnNumber: 25
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("div", { className: "m-4", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("h3", { className: "text-black text-bold text-[20px]", children: feature.name }, void 0, !1, {
            fileName: "app/Components/features.jsx",
            lineNumber: 62,
            columnNumber: 29
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime5.jsxDEV)("p", { className: "text-gray-900 text-[15px]", children: feature.decription }, void 0, !1, {
            fileName: "app/Components/features.jsx",
            lineNumber: 65,
            columnNumber: 29
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/features.jsx",
          lineNumber: 61,
          columnNumber: 25
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/features.jsx",
        lineNumber: 57,
        columnNumber: 21
      }, this)),
      ";"
    ] }, void 0, !0, {
      fileName: "app/Components/features.jsx",
      lineNumber: 54,
      columnNumber: 9
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/features.jsx",
    lineNumber: 42,
    columnNumber: 5
  }, this);
}

// app/Components/support.jsx
var import_react9 = require("react"), import_react10 = require("@material-tailwind/react"), import_outline = require("@heroicons/react/24/outline"), import_jsx_dev_runtime6 = require("react/jsx-dev-runtime"), contacts = [
  {
    name: "Sales",
    dectription: "some sales stuff that should come here. Details will folow later",
    cell: "+263 771 111 111",
    icon: import_outline.ChatBubbleBottomCenterIcon
  },
  {
    name: "Bug Report",
    dectription: "some Bug stuff that should come here. Details will folow later",
    cell: "+263 771 111 111",
    icon: import_outline.BugAntIcon
  },
  {
    name: "Tech Support",
    dectription: "some sales tech that should come here. Details will folow later",
    cell: "+263 771 111 111",
    icon: import_outline.ComputerDesktopIcon
  }
];
function Support() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("div", { id: "contact", className: "bg-white grid grid-rows-auto min-h-[100vh]  p-4", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("div", { className: "m-6 mx-auto", children: /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)(
      import_react10.Typography,
      {
        variant: "h1",
        color: "white",
        className: "mb-6 text-[#05386b] md:text-4xl lg:text-5xl",
        children: "Support Center"
      },
      void 0,
      !1,
      {
        fileName: "app/Components/support.jsx",
        lineNumber: 36,
        columnNumber: 13
      },
      this
    ) }, void 0, !1, {
      fileName: "app/Components/support.jsx",
      lineNumber: 35,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("div", { className: "grid grid-cols-1 gap-6 mx-auto lg:p-4", children: contacts.map((contact) => /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("div", { className: "flex gap-6 mb-8", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("div", { children: /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)(contact.icon, { className: "text-white w-10 h-10  bg-[#05386b] p-2 rounded" }, void 0, !1, {
        fileName: "app/Components/support.jsx",
        lineNumber: 65,
        columnNumber: 29
      }, this) }, void 0, !1, {
        fileName: "app/Components/support.jsx",
        lineNumber: 64,
        columnNumber: 25
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("div", { className: "flex flex-col", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)(
          import_react10.Typography,
          {
            variant: "lead",
            className: "text-[#5cbdb9] text-bold",
            children: contact.name
          },
          void 0,
          !1,
          {
            fileName: "app/Components/support.jsx",
            lineNumber: 68,
            columnNumber: 29
          },
          this
        ),
        /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("p", { className: "opacity-80", children: contact.dectription }, void 0, !1, {
          fileName: "app/Components/support.jsx",
          lineNumber: 75,
          columnNumber: 29
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)("p", { className: "flex mt-2", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime6.jsxDEV)(import_outline.PhoneIcon, { className: "w5 h-5 pr-1" }, void 0, !1, {
            fileName: "app/Components/support.jsx",
            lineNumber: 77,
            columnNumber: 33
          }, this),
          contact.cell
        ] }, void 0, !0, {
          fileName: "app/Components/support.jsx",
          lineNumber: 76,
          columnNumber: 29
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/support.jsx",
        lineNumber: 67,
        columnNumber: 25
      }, this)
    ] }, void 0, !0, {
      fileName: "app/Components/support.jsx",
      lineNumber: 63,
      columnNumber: 21
    }, this)) }, void 0, !1, {
      fileName: "app/Components/support.jsx",
      lineNumber: 59,
      columnNumber: 9
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/support.jsx",
    lineNumber: 34,
    columnNumber: 5
  }, this);
}

// app/Components/navbar.jsx
var import_react11 = require("@remix-run/react"), import_react12 = __toESM(require("react")), import_react13 = require("@material-tailwind/react"), import_outline2 = require("@heroicons/react/24/outline"), import_jsx_dev_runtime7 = require("react/jsx-dev-runtime");
function NavList() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)("ul", { className: "my-2 text-black flex flex-col gap-2 lg:mb-0 lg:mt-0 lg:flex-row lg:items-center lg:gap-6", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
      import_react13.Typography,
      {
        as: "li",
        variant: "small",
        color: "blue-gray",
        className: "p-1 font-medium",
        children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react11.NavLink, { to: "#about", className: "flex items-center hover:text-blue-500 transition-colors", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.InformationCircleIcon, { className: "w-4 h-4 mr-1" }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 203,
            columnNumber: 11
          }, this),
          " About"
        ] }, void 0, !0, {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 202,
          columnNumber: 9
        }, this)
      },
      void 0,
      !1,
      {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 196,
        columnNumber: 7
      },
      this
    ),
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
      import_react13.Typography,
      {
        as: "li",
        variant: "small",
        color: "blue-gray",
        className: "p-1 font-medium",
        children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react11.NavLink, { to: "#features", className: "flex items-center hover:text-blue-500 transition-colors", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.Square3Stack3DIcon, { className: "w-4 h-4 mr-1" }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 213,
            columnNumber: 10
          }, this),
          "  Features"
        ] }, void 0, !0, {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 212,
          columnNumber: 9
        }, this)
      },
      void 0,
      !1,
      {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 206,
        columnNumber: 7
      },
      this
    ),
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
      import_react13.Typography,
      {
        as: "li",
        variant: "small",
        color: "blue-gray",
        className: "p-1 font-medium",
        children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react11.NavLink, { to: "#pricing", className: "flex items-center hover:text-blue-500 transition-colors", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.CreditCardIcon, { className: "w-4 h-4 mr-1" }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 223,
            columnNumber: 10
          }, this),
          " Pricing"
        ] }, void 0, !0, {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 222,
          columnNumber: 9
        }, this)
      },
      void 0,
      !1,
      {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 216,
        columnNumber: 7
      },
      this
    ),
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
      import_react13.Typography,
      {
        as: "li",
        variant: "small",
        color: "blue-gray",
        className: "p-1 font-medium",
        children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react11.NavLink, { to: "#contact", className: "flex items-center hover:text-blue-500 transition-colors", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.PhoneIcon, { className: "w-4 h-4 mr-1" }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 233,
            columnNumber: 10
          }, this),
          " Contact Us"
        ] }, void 0, !0, {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 232,
          columnNumber: 9
        }, this)
      },
      void 0,
      !1,
      {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 226,
        columnNumber: 7
      },
      this
    ),
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
      import_react13.Typography,
      {
        as: "li",
        variant: "small",
        color: "blue-gray",
        className: "p-1 font-medium",
        children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react11.NavLink, { to: "#", className: "flex items-center hover:text-blue-500 transition-colors", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.UserCircleIcon, { className: "w-4 h-4 mr-1" }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 244,
            columnNumber: 10
          }, this),
          " Account"
        ] }, void 0, !0, {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 243,
          columnNumber: 9
        }, this)
      },
      void 0,
      !1,
      {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 237,
        columnNumber: 7
      },
      this
    )
  ] }, void 0, !0, {
    fileName: "app/Components/navbar.jsx",
    lineNumber: 195,
    columnNumber: 5
  }, this);
}
function NavBar() {
  let [openNav, setOpenNav] = import_react12.default.useState(!1), handleWindowResize = () => window.innerWidth >= 960 && setOpenNav(!1);
  return import_react12.default.useEffect(() => (window.addEventListener("resize", handleWindowResize), () => {
    window.removeEventListener("resize", handleWindowResize);
  }), []), /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react13.Navbar, { variant: "gradient", color: "black", className: "fixed z-10 w-full mx-auto px-6 py-3 rounded-none", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)("div", { className: "flex items-center justify-between text-black", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
        import_react13.Typography,
        {
          variant: "h6",
          className: "mr-4 cursor-pointer py-1.5",
          children: "Medical"
        },
        void 0,
        !1,
        {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 268,
          columnNumber: 9
        },
        this
      ),
      /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)("div", { className: "hidden lg:block", children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(NavList, {}, void 0, !1, {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 275,
        columnNumber: 11
      }, this) }, void 0, !1, {
        fileName: "app/Components/navbar.jsx",
        lineNumber: 274,
        columnNumber: 9
      }, this),
      /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(
        import_react13.IconButton,
        {
          variant: "text",
          className: "ml-auto h-6 w-6 text-inherit hover:bg-transparent focus:bg-transparent active:bg-transparent lg:hidden",
          ripple: !1,
          onClick: () => setOpenNav(!openNav),
          children: openNav ? /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.XMarkIcon, { className: "h-6 w-6", strokeWidth: 2 }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 284,
            columnNumber: 13
          }, this) : /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_outline2.Bars3Icon, { className: "h-6 w-6", strokeWidth: 2 }, void 0, !1, {
            fileName: "app/Components/navbar.jsx",
            lineNumber: 286,
            columnNumber: 13
          }, this)
        },
        void 0,
        !1,
        {
          fileName: "app/Components/navbar.jsx",
          lineNumber: 277,
          columnNumber: 9
        },
        this
      )
    ] }, void 0, !0, {
      fileName: "app/Components/navbar.jsx",
      lineNumber: 267,
      columnNumber: 7
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(import_react13.Collapse, { open: openNav, children: /* @__PURE__ */ (0, import_jsx_dev_runtime7.jsxDEV)(NavList, {}, void 0, !1, {
      fileName: "app/Components/navbar.jsx",
      lineNumber: 291,
      columnNumber: 9
    }, this) }, void 0, !1, {
      fileName: "app/Components/navbar.jsx",
      lineNumber: 290,
      columnNumber: 7
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/navbar.jsx",
    lineNumber: 266,
    columnNumber: 5
  }, this);
}

// app/Components/team.jsx
var import_react14 = require("react"), import_react15 = require("@material-tailwind/react"), import_bs = require("react-icons/bs"), import_jsx_dev_runtime8 = require("react/jsx-dev-runtime"), teams = [
  {
    name: "Progress Munoriarwa",
    position: "Co-founder / Developer",
    image: "./images/me.jpg"
  },
  {
    name: "Genius Chinembiri",
    position: "Co-founder / Developer",
    image: "./images/gc.jpg"
  },
  {
    name: "Grace Rabbit",
    position: "Senior Developer",
    image: "./images/tr.jpg"
  }
];
function Team() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)("div", { className: "grid grid-rows-auto min-h-[100vh]  p-4", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)("div", { className: "m-6 mx-auto text-center", children: [
      /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(
        import_react15.Typography,
        {
          variant: "h1",
          color: "white",
          className: "mb-6 text-white md:text-4xl lg:text-5xl",
          children: "Meet our team"
        },
        void 0,
        !1,
        {
          fileName: "app/Components/team.jsx",
          lineNumber: 33,
          columnNumber: 13
        },
        this
      ),
      /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(
        import_react15.Typography,
        {
          variant: "lead",
          color: "white",
          className: "mb-6 opacity-80",
          children: "We\u2019re a dynamic group of individuals who are passionate about what we do."
        },
        void 0,
        !1,
        {
          fileName: "app/Components/team.jsx",
          lineNumber: 40,
          columnNumber: 13
        },
        this
      )
    ] }, void 0, !0, {
      fileName: "app/Components/team.jsx",
      lineNumber: 32,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)("div", { className: "grid grid-cols-1 mx-auto gap-6 lg:grid-cols-3", children: teams.map((team) => (
      // <div className='bg-[#434757] flex flex-col h-[60vh] rounded lg: w-[25vw]'>
      //     <div className='h-3/4 px-10 py-4'>
      //         <img src={team.image} className='h-full w-full rounded-[100%] p-4' alt="" />
      //     </div>
      //     <div className='bg-yellow-50 text-center'>
      //         <h3>{team.name}</h3>
      //         <p>{team.position}</p>
      //     </div>
      // </div>
      /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.Card, { className: "w-80 bg-[#05386b] shadow-sm", children: [
        /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.CardHeader, { floated: !1, className: "h-60", children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)("img", { src: team.image, alt: "profile-picture" }, void 0, !1, {
          fileName: "app/Components/team.jsx",
          lineNumber: 63,
          columnNumber: 29
        }, this) }, void 0, !1, {
          fileName: "app/Components/team.jsx",
          lineNumber: 62,
          columnNumber: 25
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.CardBody, { className: "text-center", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.Typography, { variant: "h4", color: "white", className: "mb-2 opacity-80", children: team.name }, void 0, !1, {
            fileName: "app/Components/team.jsx",
            lineNumber: 66,
            columnNumber: 29
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.Typography, { className: "font-medium text-[#db4ddb]", textGradient: !0, children: team.position }, void 0, !1, {
            fileName: "app/Components/team.jsx",
            lineNumber: 69,
            columnNumber: 29
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/team.jsx",
          lineNumber: 65,
          columnNumber: 25
        }, this),
        /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.CardFooter, { className: "flex justify-center gap-7 pt-2", children: [
          /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.Tooltip, { content: "Like", children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(
            import_react15.Typography,
            {
              as: "a",
              href: "#facebook",
              variant: "lead",
              textGradient: !0,
              children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_bs.BsFacebook, { className: "text-[#070717]" }, void 0, !1, {
                fileName: "app/Components/team.jsx",
                lineNumber: 81,
                columnNumber: 33
              }, this)
            },
            void 0,
            !1,
            {
              fileName: "app/Components/team.jsx",
              lineNumber: 75,
              columnNumber: 29
            },
            this
          ) }, void 0, !1, {
            fileName: "app/Components/team.jsx",
            lineNumber: 74,
            columnNumber: 29
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.Tooltip, { content: "Follow", children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(
            import_react15.Typography,
            {
              as: "a",
              href: "#twitter",
              variant: "lead",
              textGradient: !0,
              children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_bs.BsTwitter, { className: "text-blue-400" }, void 0, !1, {
                fileName: "app/Components/team.jsx",
                lineNumber: 91,
                columnNumber: 33
              }, this)
            },
            void 0,
            !1,
            {
              fileName: "app/Components/team.jsx",
              lineNumber: 85,
              columnNumber: 29
            },
            this
          ) }, void 0, !1, {
            fileName: "app/Components/team.jsx",
            lineNumber: 84,
            columnNumber: 29
          }, this),
          /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_react15.Tooltip, { content: "Follow", children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(
            import_react15.Typography,
            {
              as: "a",
              href: "#instagram",
              variant: "lead",
              textGradient: !0,
              children: /* @__PURE__ */ (0, import_jsx_dev_runtime8.jsxDEV)(import_bs.BsInstagram, { className: "text-[purple]" }, void 0, !1, {
                fileName: "app/Components/team.jsx",
                lineNumber: 101,
                columnNumber: 33
              }, this)
            },
            void 0,
            !1,
            {
              fileName: "app/Components/team.jsx",
              lineNumber: 95,
              columnNumber: 29
            },
            this
          ) }, void 0, !1, {
            fileName: "app/Components/team.jsx",
            lineNumber: 94,
            columnNumber: 29
          }, this)
        ] }, void 0, !0, {
          fileName: "app/Components/team.jsx",
          lineNumber: 73,
          columnNumber: 25
        }, this)
      ] }, void 0, !0, {
        fileName: "app/Components/team.jsx",
        lineNumber: 61,
        columnNumber: 21
      }, this)
    )) }, void 0, !1, {
      fileName: "app/Components/team.jsx",
      lineNumber: 48,
      columnNumber: 9
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/team.jsx",
    lineNumber: 31,
    columnNumber: 5
  }, this);
}

// app/routes/_index.jsx
var import_jsx_dev_runtime9 = require("react/jsx-dev-runtime");
function Index() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)("div", { children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)(NavBar, {}, void 0, !1, {
      fileName: "app/routes/_index.jsx",
      lineNumber: 13,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)(Slider, {}, void 0, !1, {
      fileName: "app/routes/_index.jsx",
      lineNumber: 14,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)(Features, {}, void 0, !1, {
      fileName: "app/routes/_index.jsx",
      lineNumber: 15,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)(pricing_default, {}, void 0, !1, {
      fileName: "app/routes/_index.jsx",
      lineNumber: 16,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)(Support, {}, void 0, !1, {
      fileName: "app/routes/_index.jsx",
      lineNumber: 17,
      columnNumber: 9
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime9.jsxDEV)(Team, {}, void 0, !1, {
      fileName: "app/routes/_index.jsx",
      lineNumber: 18,
      columnNumber: 9
    }, this)
  ] }, void 0, !0, {
    fileName: "app/routes/_index.jsx",
    lineNumber: 11,
    columnNumber: 5
  }, this);
}

// app/routes/notes.jsx
var notes_exports = {};
__export(notes_exports, {
  default: () => About
});

// app/Components/NewNote.jsx
var import_react17 = require("@remix-run/react"), import_jsx_dev_runtime10 = require("react/jsx-dev-runtime");
function NewNote() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)("div", { className: " create_notes", children: /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)(import_react17.Form, { method: "Post", children: [
    /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)("input", { type: "text", name: "name", placeholder: "enter patient name" }, void 0, !1, {
      fileName: "app/Components/NewNote.jsx",
      lineNumber: 8,
      columnNumber: 17
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)("br", {}, void 0, !1, {
      fileName: "app/Components/NewNote.jsx",
      lineNumber: 8,
      columnNumber: 82
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)("textarea", { name: "notes", placeholder: "enter patient notes" }, void 0, !1, {
      fileName: "app/Components/NewNote.jsx",
      lineNumber: 9,
      columnNumber: 17
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)("br", {}, void 0, !1, {
      fileName: "app/Components/NewNote.jsx",
      lineNumber: 9,
      columnNumber: 85
    }, this),
    /* @__PURE__ */ (0, import_jsx_dev_runtime10.jsxDEV)("button", { children: "create notes" }, void 0, !1, {
      fileName: "app/Components/NewNote.jsx",
      lineNumber: 10,
      columnNumber: 17
    }, this)
  ] }, void 0, !0, {
    fileName: "app/Components/NewNote.jsx",
    lineNumber: 7,
    columnNumber: 13
  }, this) }, void 0, !1, {
    fileName: "app/Components/NewNote.jsx",
    lineNumber: 6,
    columnNumber: 9
  }, this);
}

// app/routes/notes.jsx
var import_jsx_dev_runtime11 = require("react/jsx-dev-runtime");
function About() {
  return /* @__PURE__ */ (0, import_jsx_dev_runtime11.jsxDEV)("div", { className: "notes_div", children: /* @__PURE__ */ (0, import_jsx_dev_runtime11.jsxDEV)(NewNote, {}, void 0, !1, {
    fileName: "app/routes/notes.jsx",
    lineNumber: 7,
    columnNumber: 13
  }, this) }, void 0, !1, {
    fileName: "app/routes/notes.jsx",
    lineNumber: 6,
    columnNumber: 9
  }, this);
}

// server-assets-manifest:@remix-run/dev/assets-manifest
var assets_manifest_default = { entry: { module: "/build/entry.client-ACFSEPTX.js", imports: ["/build/_shared/chunk-ZWGWGGVF.js", "/build/_shared/chunk-GIAAE3CH.js", "/build/_shared/chunk-2H4SZ7M7.js", "/build/_shared/chunk-H2LASQMJ.js", "/build/_shared/chunk-NMZL6IDN.js", "/build/_shared/chunk-BOXFZXVX.js", "/build/_shared/chunk-UWV35TSL.js", "/build/_shared/chunk-PNG5AS42.js"] }, routes: { root: { id: "root", parentId: void 0, path: "", index: void 0, caseSensitive: void 0, module: "/build/root-KE7CSJKF.js", imports: void 0, hasAction: !1, hasLoader: !1, hasCatchBoundary: !1, hasErrorBoundary: !1 }, "routes/_index": { id: "routes/_index", parentId: "root", path: void 0, index: !0, caseSensitive: void 0, module: "/build/routes/_index-34LPAWDG.js", imports: void 0, hasAction: !1, hasLoader: !1, hasCatchBoundary: !1, hasErrorBoundary: !1 }, "routes/notes": { id: "routes/notes", parentId: "root", path: "notes", index: void 0, caseSensitive: void 0, module: "/build/routes/notes-U7OZJJQG.js", imports: void 0, hasAction: !1, hasLoader: !1, hasCatchBoundary: !1, hasErrorBoundary: !1 } }, version: "e3eb0abf", hmr: { runtime: "/build/_shared\\chunk-H2LASQMJ.js", timestamp: 1689114511114 }, url: "/build/manifest-E3EB0ABF.js" };

// server-entry-module:@remix-run/dev/server-build
var assetsBuildDirectory = "public\\build", future = { v2_dev: !0, unstable_postcss: !1, unstable_tailwind: !1, v2_errorBoundary: !0, v2_headers: !0, v2_meta: !0, v2_normalizeFormMethod: !0, v2_routeConvention: !0 }, publicPath = "/build/", entry = { module: entry_server_exports }, dev = { port: 3001 }, routes = {
  root: {
    id: "root",
    parentId: void 0,
    path: "",
    index: void 0,
    caseSensitive: void 0,
    module: root_exports
  },
  "routes/_index": {
    id: "routes/_index",
    parentId: "root",
    path: void 0,
    index: !0,
    caseSensitive: void 0,
    module: index_exports
  },
  "routes/notes": {
    id: "routes/notes",
    parentId: "root",
    path: "notes",
    index: void 0,
    caseSensitive: void 0,
    module: notes_exports
  }
};
// Annotate the CommonJS export names for ESM import in node:
0 && (module.exports = {
  assets,
  assetsBuildDirectory,
  dev,
  entry,
  future,
  publicPath,
  routes
});
//# sourceMappingURL=index.js.map
