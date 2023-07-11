import {
  Links,
  LiveReload,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration
} from "/build/_shared/chunk-2H4SZ7M7.js";
import {
  createHotContext
} from "/build/_shared/chunk-H2LASQMJ.js";
import {
  require_jsx_runtime
} from "/build/_shared/chunk-NMZL6IDN.js";
import "/build/_shared/chunk-BOXFZXVX.js";
import "/build/_shared/chunk-UWV35TSL.js";
import {
  __toESM
} from "/build/_shared/chunk-PNG5AS42.js";

// app/root.jsx
var import_jsx_runtime = __toESM(require_jsx_runtime());

// app/styles/main.css
var main_default = "/build/_assets/main-XQIXJFXJ.css";

// app/tailwind/tailwind.css
var tailwind_default = "/build/_assets/tailwind-KVIRLIZX.css";

// app/root.jsx
if (!window.$RefreshReg$ || !window.$RefreshSig$ || !window.$RefreshRuntime$) {
  console.warn("remix:hmr: React Fast Refresh only works when the Remix compiler is running in development mode.");
} else {
  prevRefreshReg = window.$RefreshReg$;
  prevRefreshSig = window.$RefreshSig$;
  window.$RefreshReg$ = (type, id) => {
    window.$RefreshRuntime$.register(type, '"app\\\\root.jsx"' + id);
  };
  window.$RefreshSig$ = window.$RefreshRuntime$.createSignatureFunctionForTransform;
}
var prevRefreshReg;
var prevRefreshSig;
if (import.meta) {
  import.meta.hot = createHotContext(
    //@ts-expect-error
    "app\\root.jsx"
  );
}
function App() {
  return /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("html", {
    lang: "en",
    children: [/* @__PURE__ */ (0, import_jsx_runtime.jsxs)("head", {
      children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)("meta", {
        charSet: "utf-8"
      }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)("meta", {
        name: "viewport",
        content: "width=device-width,initial-scale=1"
      }), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Meta, {}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Links, {})]
    }), /* @__PURE__ */ (0, import_jsx_runtime.jsxs)("body", {
      children: [/* @__PURE__ */ (0, import_jsx_runtime.jsx)(Outlet, {}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ScrollRestoration, {}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(Scripts, {}), /* @__PURE__ */ (0, import_jsx_runtime.jsx)(LiveReload, {})]
    })]
  });
}
_c = App;
function links() {
  return [{
    rel: "stylesheet",
    href: main_default
  }, {
    rel: "stylesheet",
    href: tailwind_default
  }];
}
var _c;
$RefreshReg$(_c, "App");
window.$RefreshReg$ = prevRefreshReg;
window.$RefreshSig$ = prevRefreshSig;
export {
  App as default,
  links
};
//# sourceMappingURL=/build/root-KE7CSJKF.js.map
