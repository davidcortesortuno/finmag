/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 *
 */

#pragma once

#include <cxxabi.h>

// TODO: test this whole thing...
namespace finmag {
    namespace util {
        namespace bp = boost::python;

        // Returns the full C++ type name for the specified type
        template<class T>
        std::string get_type_name() {
            // use RTTI and IA64 ABI to get the full class name for T
            const char *mangled_name = typeid(T).name();
            int status = -10;
            char *demangled_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);
            if (status < 0) throw std::runtime_error(std::string("Could not demangle C++ name ") + boost::lexical_cast<std::string>(mangled_name));
            std::string res(demangled_name);
            free(demangled_name);
            return res;
        }

        // Have to wrap classes using the wrapper to avoid invoking the default constuctor when iterating using for_each
        struct wrap_type {
            template<class T>
            struct wrapped_type { typedef T type; };

            template<class T>
            struct apply {
                typedef wrapped_type<T> type;
            };
        };

        // The following definitions were extracted from the SWIG .cxx files generated by a build of fenics
        // Should the SWIG definitions change, our code might fail with some SegFaults...
        typedef struct swig_type_info {
            const char             *name;			/* mangled name of this type */
            const char             *str;			/* human readable name of this type */
            // rest omitted
        } swig_type_info;

        typedef struct {
            PyObject_HEAD
            void *ptr;
            swig_type_info *ty;
            // rest omitted
        } SwigPyObject;

        // for readability, separate the code to unwrap swig objects into this function
        std::pair<void*, const char*> unwrap_swig_object(PyObject * obj) {
            SwigPyObject* swig_object = (SwigPyObject*) PyObject_GetAttrString(obj, "this");
            void *ptr = swig_object->ptr;
            const char *str = swig_object->ty->str;
            Py_DECREF(swig_object);
            return std::make_pair(ptr, str);
        }

        template<class BoostPythonType>
        struct swig_converter_algorithms {
            static void* convertible_from_python(PyObject* obj, const std::string &swig_type_string) {
                if (!PyObject_HasAttrString(obj, "this")) return 0;
                std::pair<void*, const char*> swig_obj = unwrap_swig_object(obj);
//                printf("convertible_from_python: expected '%s', provided '%s'\n", swig_type_string.c_str(), swig_obj.second);
                if (strcmp(swig_type_string.c_str(), swig_obj.second) != 0) return 0;
                return obj;
            }

            // Converts a Python ndarray to a np_array
            template <class T>
            static void construct_from_python(const T &value, bp::converter::rvalue_from_python_stage1_data* data) {
                // grab pointer to memory into which to construct the new C++ object
                void* storage = ((bp::converter::rvalue_from_python_storage<BoostPythonType>*) data)->storage.bytes;

                // construct the new C++ object in place
                new (storage) BoostPythonType(value);

                // save the memory chunk pointer for later use by boost.python
                data->convertible = storage;
            }
        };

        template<class SwigDataType>
        struct swig_shared_ptr_type_string { static std::string value; };
        template<class SwigDataType>
        std::string swig_shared_ptr_type_string<SwigDataType>::value =
                std::string("std::shared_ptr< ") + get_type_name<SwigDataType>() + " > *";

        // With a large class hierarchy the use of multiple converters may be inefficient as convertible_from_python
        // has to be executed for each derived class
        // TODO: convert this to use a std::map based on the swig type supplied
        template<class BaseClass, class DerivedClass>
        class swig_shared_ptr_derived_class_converter {
        private:
            typedef std::shared_ptr<BaseClass> boost_python_type;
            typedef std::shared_ptr<DerivedClass> swig_type;

        public:

            static void* convertible_from_python(PyObject* obj) {
                return swig_converter_algorithms<boost_python_type>::convertible_from_python(obj, swig_shared_ptr_type_string<DerivedClass>::value);
            }

            // Converts a Python ndarray to a np_array
            static void construct_from_python(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data) {
                // unwrap the swig object
                std::pair<void*, const char*> swig_obj = unwrap_swig_object(obj_ptr);
                // get the derived class shared_ptr pointer
                std::shared_ptr<DerivedClass> &derived_class_ptr = *(std::shared_ptr<DerivedClass> *) swig_obj.first;
                // cast the derived class shared_ptr pointer to a base class shared_ptr pointer
                std::shared_ptr<BaseClass> base_ptr = std::dynamic_pointer_cast<BaseClass>(derived_class_ptr);

                swig_converter_algorithms<boost_python_type>::construct_from_python(base_ptr, data);
            }

            static void register_converter() {
            // we only supply the from-python converter for now
                bp::converter::registry::push_back(&convertible_from_python, &construct_from_python, bp::type_id<boost_python_type>());
            }
        };

        template<class BaseClass>
        struct derived_class_shared_ptr_initialiser
        {
            template<class WrappedDerivedClass> void operator()(WrappedDerivedClass  x) {
                // unwrap the derived class
                typedef typename WrappedDerivedClass::type derived_class;
                // check that the derived class is actually derived from the base class
                BaseClass *p = (derived_class*)0; if (p) {}

//                printf("registering derived %s\n", get_type_name<derived_class>().c_str());
                swig_shared_ptr_derived_class_converter<BaseClass, derived_class>::register_converter();
            }
        };

        template<class BaseClass, class DerivedClassesList>
        void register_swig_boost_shared_ptr_hierarchy()  {
            // wrap derived classes to avoid invoking the default constuctor when iterating using for_each
            typedef typename boost::mpl::transform<DerivedClassesList, wrap_type>::type wrapped_derived_classes;

//            printf("registering %s\n", get_type_name<object_type>().c_str());
            boost::mpl::for_each<wrapped_derived_classes>(derived_class_shared_ptr_initialiser<BaseClass>());
            // TODO: not use the derived class converter here (?)
            swig_shared_ptr_derived_class_converter<BaseClass, BaseClass>::register_converter();
        }

        template<class DataType>
        void register_swig_boost_shared_ptr() {
            // TODO: not use the derived class converter here (?)
            swig_shared_ptr_derived_class_converter<DataType, DataType>::register_converter();
        }
    }
}